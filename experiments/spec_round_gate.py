"""Phase 5 gate — rejection sampling and bookkeeping inside the engine.

Phase 4 proved the verify pass computes what K+1 sequential decode steps would.
Phase 5 turns that into a decode round: the draft proposes K tokens, the target
verifies them in one pass, rejection sampling decides how many survive, and the
scheduler commits between 1 and K+1 tokens per request. This script gates the
new parts of that, in increasing order of integration.

**Gate 1 — the ported sampler still returns the target's distribution.**
`Sampler.rejection_sample` is the Phase 2 prototype's pure function moved into
`minivllm/models/layers/sampler.py` and batched across requests. The batching is
not a cosmetic change — accept/reject decisions are now made for the whole batch
in one shot — so the Leviathan/Chen lemma is re-checked on the engine's own
function: over many trials against deliberately mismatched p and q, the first
token out of a round must be distributed exactly as p, whatever q is. No models
and no GPU involved.

**Gate 2 — a round emits exactly what greedy decoding would have.** Acceptance
is scripted rather than left to the draft, because a random-init draft is
rejected at i=0 every single round and would leave partial and full acceptance
untested. At an anchor, the K+1 tokens plain decoding produces are recorded as
ground truth, proposals are built with exactly M of them correct, and the round
is run on the engine's real `verify()` output. Two things must hold:
`num_accepted == M`, and the emitted tokens are `truth[:M+1]` — the accepted
prefix plus one, whether that one came from the residual resample or from the
bonus row. M is swept over 0..K, so the zero-acceptance case that implementations
get wrong first is covered at every anchor.

A mismatch is a hard failure unless that step's own top-2 logit gap is below the
batch-shape noise floor, i.e. a genuine tie that a sub-ULP perturbation may
reorder. Those are counted and reported, never hidden. This is the same
classification `experiments/verify_pass_gate.py` uses, and the reason is
unchanged: a verify pass runs at a different batch width than a decode step, and
bf16 GEMM tiling moves logits by up to 0.5 absolute on that alone.

**Gate 3 — the whole engine, end to end.** `Engine.generate` with speculation on
versus off, greedy, same prompts. Every round is checked for the invariant that
makes forward progress guaranteed (`len(tokens) == num_accepted + 1`, so a round
can never emit zero tokens even when the draft is useless), the executor is
checked for not having committed anything itself, the blocks backing the
proposal region are checked, and the acceptance rate is reported.

It runs at both extremes of acceptance, because the two exercise different code:

  - the **random draft**, rejected at i=0 in every round, so every round takes
    the residual-resample path and leaves the whole proposal region stale;
  - the **target drafting for itself**, which accepts all K in every round.
    This is the only regime where the draft finishes a round one token behind
    the committed sequence, so it is the only one that exercises the catch-up
    row `Executor.propose` prepends — if that row were wrong, the draft would be
    proposing from a stale cache and acceptance would fall off 100% after the
    first round. It needs VRAM for two copies of the target and is skipped, with
    a message, when there is not enough.

Usage:
    python experiments/spec_round_gate.py
    python experiments/spec_round_gate.py --k 8
    python experiments/spec_round_gate.py --no-self-draft
"""

import argparse
import logging
import os
import sys

# Must precede the torch import: inductor reads this at config-module import.
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minivllm.config.sampling import SamplingParams  # noqa: E402
from minivllm.engine.engine import Engine  # noqa: E402
from minivllm.models.layers.sampler import Sampler  # noqa: E402

# The primitives for driving the executor by hand live with the regression
# suite, which is their durable home; this script is one of two consumers. See
# tests/spec_harness.py.
from tests.spec_harness import (  # noqa: E402
    DRAFT,
    PROMPTS,
    TARGET,
    batch_shape_noise_floor,
    build_engine,
    chat_prompts,
    free_gpu_memory,
    prefill_one,
    reference_walk,
    release,
    run_round,
    scripted_proposal,
    sequential_steps,
)

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# gate 1 — the maths, no models and no GPU
# ---------------------------------------------------------------------------

def gate_sampler_distribution(k: int, trials: int, seed: int) -> bool:
    """The first token out of a round must be distributed exactly as p.

    Runs the engine's `Sampler.rejection_sample`, not a re-derivation of it, and
    runs every trial as one row of a single batched call — which is also the
    only direct test that batching did not break the per-request independence
    the maths assumes.
    """
    print("\n== gate 1: Sampler.rejection_sample reproduces the target distribution ==")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampler = Sampler()
    vocab = 8
    ok = True

    for num_draft in (1, k):
        # Deliberately mismatched p and q, with a token q loves and p does not:
        # the draft will keep proposing token 0 and it must keep being rejected.
        p = torch.rand(num_draft + 1, vocab, generator=generator) + 0.05
        q = torch.rand(num_draft, vocab, generator=generator) + 0.05
        q[:, 0] += 3.0
        p[:, 0] = 0.01
        p /= p.sum(-1, keepdim=True)
        q /= q.sum(-1, keepdim=True)

        draft_tokens = torch.multinomial(q, trials, replacement=True,
                                         generator=generator).T.contiguous()
        batched_q = q.unsqueeze(0).expand(trials, -1, -1)
        batched_p = p.unsqueeze(0).expand(trials, -1, -1)

        results = sampler.rejection_sample(draft_tokens, batched_q, batched_p, generator)

        counts = torch.zeros(vocab)
        for tokens, num_accepted in results:
            assert len(tokens) == num_accepted + 1, (
                f"a round returned {len(tokens)} tokens for {num_accepted} accepted; "
                f"the invariant is num_accepted + 1"
            )
            counts[tokens[0]] += 1

        empirical = counts / trials
        expected = p[0]
        stderr = (expected * (1 - expected) / trials).sqrt()
        worst = float(((empirical - expected).abs() / stderr).max())
        good = worst < 5.0
        ok &= good

        print(f"   K={num_draft}: max |empirical - p| = {float((empirical - expected).abs().max()):.4f} "
              f"({worst:.2f} standard errors over {trials} trials)  "
              f"{'ok' if good else 'FAILED'}")

    print(f"   gate 1 {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# gate 2 — a round with scripted acceptance
# ---------------------------------------------------------------------------

def gate_scripted_acceptance(engine: Engine, prompt_tokens: list[int], reference: list[int],
                             k: int, floor: float, label: str) -> tuple[int, int, int]:
    """Sweep M over 0..K and check the round emits exactly `truth[:M+1]`.

    Returns (rounds_checked, hard_failures, tolerated_ties).
    """
    vocab_size = engine.config.hf_config.vocab_size
    anchors = [g for g in (1, 4, 12, 24, 36, 48, 60) if g < len(reference)]

    checked = hard = ties = 0

    for num_correct in range(k + 1):
        script_hard = script_ties = 0

        for g in anchors:
            prefix = prompt_tokens + reference[:g]

            # Ground truth: what K+1 plain decode steps produce from a fresh
            # prefill of this exact prefix. truth[j] is the token that follows
            # query row j of the verify pass, so proposal j is "correct" — and
            # must be accepted — precisely when it equals truth[j].
            truth, _, gaps = sequential_steps(engine, prefix, k + 1)

            proposal = scripted_proposal(truth, num_correct, k, vocab_size)

            req, _ = prefill_one(engine, prefix)
            tokens, num_accepted = run_round(engine, req, proposal)
            release(engine, req)

            checked += 1

            assert len(tokens) == num_accepted + 1, (
                f"round returned {len(tokens)} tokens for {num_accepted} accepted"
            )

            expected_tokens = truth[:num_correct + 1]
            if num_accepted == num_correct and tokens == expected_tokens:
                continue

            # A divergence is only forgivable where the target itself was
            # within noise of a tie at the step that went wrong.
            first_bad = next((j for j in range(min(len(tokens), len(expected_tokens)))
                              if tokens[j] != expected_tokens[j]), min(num_accepted, num_correct))
            if gaps[min(first_bad, len(gaps) - 1)] <= floor:
                script_ties += 1
                continue

            script_hard += 1
            print(f"     MISMATCH {label} M={num_correct} anchor g={g}: "
                  f"accepted {num_accepted} (want {num_correct}), "
                  f"emitted {tokens} (want {expected_tokens}), "
                  f"top-2 gap at step {first_bad} = {gaps[min(first_bad, len(gaps) - 1)]:.3f} "
                  f"> floor {floor:.3f}")

        hard += script_hard
        ties += script_ties
        regime = ("zero" if num_correct == 0 else
                  "full" if num_correct == k else "partial")
        print(f"     M={num_correct} ({regime:7s}): {len(anchors)} anchors, "
              + ("clean" if script_hard == 0 else f"{script_hard} HARD FAILURES")
              + (f", {script_ties} near-tie" if script_ties else ""))

    return checked, hard, ties


# ---------------------------------------------------------------------------
# gate 3 — end to end through Engine.generate
# ---------------------------------------------------------------------------

class RoundRecorder:
    """Wraps `Executor.execute_speculative` to check every round's invariants."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.k = engine.config.num_speculative_tokens
        self.rounds = 0
        self.emitted = 0
        self.proposed = 0
        self.accepted = 0
        self.acceptance_histogram = [0] * (self.k + 1)
        self.violations: list[str] = []

        self._inner = engine.executor.execute_speculative
        engine.executor.execute_speculative = self._record

    def detach(self):
        """Drop every reference to the engine, keeping only the counters.

        `self._inner` is a *bound* method, so holding it pins the executor, the
        model weights and the KV cache. Without this the engines built earlier in
        the run are never collected and the last one has no memory left to size a
        cache against.
        """
        self.engine.executor.execute_speculative = self._inner
        self._inner = None
        self.engine = None

    def _record(self, batch):
        before = [len(req.tokens) for req in batch.requests]
        tokens, num_accepted, num_proposed = self._inner(batch)
        # 0 means the proposer declined and the step fell back to plain decode
        # (n-gram only). Such a step proposed nothing, so counting K per request
        # would understate acceptance.
        per_request = num_proposed // len(batch.requests) if num_proposed else 0

        for req, length, emitted, accepted in zip(batch.requests, before, tokens, num_accepted):
            self.rounds += 1
            self.emitted += len(emitted)
            self.proposed += per_request
            self.accepted += accepted
            self.acceptance_histogram[accepted] += 1

            if len(emitted) != accepted + 1:
                self.violations.append(
                    f"request {req.id}: emitted {len(emitted)} tokens for {accepted} accepted")
            if not 0 <= accepted <= self.k:
                self.violations.append(f"request {req.id}: accepted {accepted} of {self.k}")
            # The round must not have touched the request; committing is the
            # scheduler's job, and double-committing would silently duplicate
            # tokens.
            if len(req.tokens) != length:
                self.violations.append(
                    f"request {req.id}: execute_speculative changed req.tokens")
            # Every position the round wrote KV for has to be backed by a block.
            needed = -(-(length + self.k) // self.engine.config.kv_cache_block_size)
            if len(req.blocks) < needed:
                self.violations.append(
                    f"request {req.id}: {len(req.blocks)} blocks, needs {needed} for "
                    f"{length} tokens + {self.k} proposals")

        return tokens, num_accepted, num_proposed


def run_engine(spec: bool, draft: str, k: int, max_tokens: int, prompts: list[list[int]]):
    engine = build_engine(spec=spec, draft=draft, k=k)
    recorder = RoundRecorder(engine) if spec else None

    # temperature 1.0 with top_k 0 and top_p 1.0 is Sampler.forward's argmax
    # branch: deterministic greedy, and no flashinfer import.
    sp = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, max_tokens=max_tokens)
    outputs = engine.generate(prompts, sp, use_tqdm=False)
    stats = engine.metrics.stats()

    result = ([o["tokens"] for o in outputs], recorder, stats)

    # nn.Module graphs hold reference cycles, so refcounting alone will not free
    # the weights; without the collect a second engine on a 4 GB card sizes its
    # cache against memory the first one is still holding.
    if recorder is not None:
        recorder.detach()
    del engine
    free_gpu_memory()
    return result


def gate_end_to_end(draft: str, k: int, max_tokens: int, prompts: list[list[int]],
                    label: str) -> bool:
    print(f"\n== gate 3: Engine.generate, speculation on vs off ({label}) ==")

    base_tokens, _, _ = run_engine(False, draft, k, max_tokens, prompts)
    spec_tokens, recorder, stats = run_engine(True, draft, k, max_tokens, prompts)

    ok = True
    for i, (want, got) in enumerate(zip(base_tokens, spec_tokens)):
        same = want == got
        ok &= same
        print(f"   [{i}] {len(want):3d} baseline tokens, {len(got):3d} speculative, "
              f"identical={same}")
        if not same:
            first = next((j for j, (a, b) in enumerate(zip(want, got)) if a != b),
                         min(len(want), len(got)))
            print(f"        first divergence at token {first}: "
                  f"{want[first:first + 3]} vs {got[first:first + 3]}")

    print(f"\n   {recorder.rounds} rounds, {recorder.emitted} tokens emitted, "
          f"{recorder.accepted}/{recorder.proposed} proposals accepted "
          f"({stats.acceptance_rate:.1%})")
    print(f"   tokens per request per decode step: {stats.tokens_per_request_step:.2f}  "
          f"(1.00 means the draft never landed; {k + 1:.2f} is the ceiling)")
    print("   acceptance histogram: "
          + "  ".join(f"M={m}:{n}" for m, n in enumerate(recorder.acceptance_histogram) if n))

    if recorder.violations:
        ok = False
        print(f"   {len(recorder.violations)} ROUND INVARIANT VIOLATIONS:")
        for violation in recorder.violations[:10]:
            print(f"     {violation}")
    else:
        print("   round invariants hold in every round "
              "(emitted == accepted + 1, blocks cover the proposal region, "
              "req.tokens untouched by the executor)")

    print(f"   gate 3 {'PASSED' if ok else 'FAILED'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="proposals per round")
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--anchor-tokens", type=int, default=64,
                    help="length of the reference walk gate 2 anchors along")
    ap.add_argument("--trials", type=int, default=20000,
                    help="samples for the distribution gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-self-draft", dest="self_draft", action="store_false",
                    help="skip the 100%%-acceptance run (needs VRAM for two targets)")
    args = ap.parse_args()

    args.draft = os.path.expanduser(args.draft)

    ok1 = gate_sampler_distribution(args.k, args.trials, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    prompts = chat_prompts(tokenizer)

    # Gate 2 drives the executor by hand and never needs the draft loaded.
    engine = build_engine(k=args.k)
    eos_ids = set(engine.config.eos_token_ids or set())
    floor = batch_shape_noise_floor(engine, prompts[0], args.k + 1, verbose=True)

    print("\n== gate 2: a round emits exactly what greedy decoding would ==")
    checked_total = hard_total = tie_total = 0
    for i, prompt_tokens in enumerate(prompts):
        reference = reference_walk(engine, prompt_tokens, args.anchor_tokens, eos_ids)
        print(f"\n   prompt {i}: {tokenizer.decode(prompt_tokens)[-42:]!r} "
              f"-> {len(reference)} reference tokens")
        checked, hard, ties = gate_scripted_acceptance(
            engine, prompt_tokens, reference, args.k, floor, f"p{i}")
        checked_total += checked
        hard_total += hard
        tie_total += ties

    ok2 = hard_total == 0
    print(f"\n   {checked_total} rounds checked across M=0..{args.k}, "
          f"{hard_total} hard mismatches, {tie_total} near-tie reorderings tolerated")
    print(f"   gate 2 {'PASSED' if ok2 else f'FAILED ({hard_total} hard mismatches)'}")

    del engine
    free_gpu_memory()

    ok3 = gate_end_to_end(args.draft, args.k, args.max_tokens, prompts,
                          "random draft: rejection at i=0 every round")

    # The random draft never lands a proposal, so end to end it only ever
    # exercises the zero-acceptance path. Running the target as its own draft is
    # the cheap way to reach the other extreme with one checkpoint: greedy
    # against greedy accepts all K every round, which is also the only regime
    # where the draft ends a round behind and its catch-up row does real work.
    # If that row were wrong, the draft would propose from a stale cache and
    # acceptance would fall off 100% after the first round.
    ok4 = True
    if args.self_draft:
        try:
            ok4 = gate_end_to_end(TARGET, args.k, args.max_tokens, prompts,
                                  "self-draft: full acceptance every round")
        except AssertionError as exc:
            if "No memory left for the KV cache" not in str(exc):
                raise
            print("\n== gate 3b SKIPPED: not enough VRAM for two copies of the target ==")
            print(f"   {exc}")
            print("   Re-run with --no-self-draft to silence this, or on a larger GPU.")

    all_ok = ok1 and ok2 and ok3 and ok4
    print("\nGATE:", "PASS - speculative rounds reproduce greedy decoding"
          if all_ok else "FAIL - see above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
