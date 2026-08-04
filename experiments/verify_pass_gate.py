"""Phase 4 gate — the multi-token verify pass equals sequential decoding.

Phase 4's contract: one forward pass over K+1 query tokens per request must
compute what K+1 one-token decode steps would have computed, against the same
paged KV cache. Rejection sampling is Phase 5 and is not exercised here beyond
its greedy degenerate form.

A word on what "equals" can mean here, because it is not "byte-identical" and
the reason matters for Phase 6.

The model's logits depend on the batch shape it is run at. Decoding one request
at batch size 1 and at batch size 5 — no verify pass involved at all, just
`_build_decode_input` twice — gives logits differing by up to 0.5 absolute
(mean 0.087) on Qwen3-0.6B in bf16, because cuBLAS picks different tilings and
`@torch.compile` emits a different MLP kernel per shape. Within one shape it is
perfectly deterministic (five identical rows in one batch agree to 0.0000).
A verify pass changes the batch shape by construction: K+1 rows where decode
has 1. So its logits *cannot* be bitwise equal to sequential decode's, no
matter how correct it is, and any gate demanding that is measuring bf16 GEMM
tiling rather than speculative decoding.

`batch_shape_noise_floor` measures that floor, and the gates below are stated
relative to it:

**Gate 1 — context construction is exact.** `verify()` with zero proposals is a
single query row, the same shape as a decode step, so here the numerics
argument does not apply and the two must agree *bitwise*. This pins down
`positions`, `cache_seqlens`, `slot_mapping` and `block_table` with no slack.

**Gate 2 — the verify pass adds no error of its own.** verify()'s row 0 versus a
decode step must deviate by no more than the batch-shape control does. If the
verify pass were subtly wrong, its deviation would exceed the floor.

**Gate 3 — one verify pass equals K+1 sequential decode steps.** This is the
literal contract, and it is a *short-horizon* claim, which matters. The KV cache
is not only read by these passes, it is written by them, so a batch-1 decode
walk and a batch-(K+1) verify walk lay down subtly different K/V values that
compound. Over sixty rounds the two trajectories drift far enough apart to flip
a confident argmax, which says nothing about whether the verify pass is right.

So each measurement is freshly anchored. At an anchor `g`, two independent
requests are prefilled on the identical token prefix `prompt + reference[:g]`,
giving both an identical, prefill-built cache. One is advanced with K+1 plain
decode steps; the other runs a single verify pass. Their logits and argmaxes are
compared over that K+1-step horizon only, then both are discarded. Anchors are
spread along the reference so the test covers short and long contexts and both
sides of a block boundary.

Acceptance regimes are scripted on top of each anchor by corrupting proposals:

  - `oracle`  — all K proposals correct: full acceptance, K+1 tokens.
  - `zero`    — all wrong: rejection at j=0, and the entire proposal region goes
                stale. The case implementations get wrong first.
  - `partial` — M correct then a wrong one.
  - `mixed`   — M varies with the anchor, covering every stale-region size.

Only rows 0..M are checked: a row after the first corrupted proposal follows a
prefix that never existed, so it legitimately predicts something else. A
mismatch is a hard failure unless that step's own top-2 logit gap is below the
noise floor, i.e. a genuine tie a sub-ULP perturbation may reorder. Those are
counted and reported, never hidden.

**A note on K.** A verify pass runs at batch width `num_requests * (K+1)`. This
gate was originally capped at K=4 because the decode kernel was nondeterministic
at width 6 and above; that was a backend race, since fixed by
`patches/mini-flash-attention-decode-race.patch`, and K is no longer bounded by
it. Verified clean at K=4, 8 and 16. If a run at higher K starts reporting hard
mismatches, check `experiments/decode_determinism_check.py` first — that is the
failure mode returning, not the verify pass breaking.

Usage:
    python experiments/verify_pass_gate.py
    python experiments/verify_pass_gate.py --k 8
    python experiments/verify_pass_gate.py --k 16 --max-tokens 96
"""

import argparse
import logging
import os
import sys

# Must precede the torch import: inductor reads this at config-module import.
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer  # noqa: E402

from minivllm.engine.engine import Engine  # noqa: E402

# The primitives for driving the executor by hand live with the regression
# suite, which is their durable home; this script is one of two consumers. See
# tests/spec_harness.py.
from tests.spec_harness import (  # noqa: E402
    PROMPTS,
    TARGET,
    batch_shape_noise_floor,
    build_config,
    chat_prompts,
    decode_logits,
    prefill_one,
    reference_walk,
    release,
    scripted_proposal,
    sequential_steps,
)

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# gate 1 + 2
# ---------------------------------------------------------------------------

def gate_context_exact(engine: Engine, prompt_tokens: list[int]) -> bool:
    """verify() with no proposals is one query row: it must match decode bitwise."""
    print("\n== gate 1: verify() with K=0 == plain decode, bitwise ==")
    req, _ = prefill_one(engine, prompt_tokens)
    expected = decode_logits(engine, [req])[0]
    got = engine.executor.verify([req], [[]])[0, 0].float()
    max_diff = float((expected - got).abs().max())
    release(engine, req)

    ok = max_diff == 0.0
    print(f"   max|decode - verify| = {max_diff:.6f}   (must be exactly 0)")
    print(f"   gate 1 {'PASSED' if ok else 'FAILED'}")
    return ok


def gate_no_added_error(engine: Engine, prompt_tokens: list[int], k: int) -> bool:
    """verify()'s row 0 must add no error beyond running at width K+1.

    The floor is measured at *this same state* rather than borrowed from
    elsewhere, since it scales with the logit magnitudes at the position in
    question. Decoding this state at width 1 and at width K+1 brackets how much
    the batch shape alone moves things; the verify pass, which also runs at
    width K+1, must land inside that bracket.
    """
    print("\n== gate 2: verify() row 0 adds no error beyond its batch width ==")
    requests = [prefill_one(engine, prompt_tokens)[0] for _ in range(k + 1)]

    single = decode_logits(engine, requests[:1])[0]
    batched = decode_logits(engine, requests)[0]
    state_floor = float((single - batched).abs().max())

    vocab_size = engine.config.hf_config.vocab_size
    proposal = [(i + 7) % vocab_size for i in range(k)]
    engine.scheduler.block_manager.allocate_block_for_decode(requests[0], extra_tokens=k)
    got = engine.executor.verify([requests[0]], [proposal])[0, 0].float()

    vs_single = float((single - got).abs().max())
    vs_batched = float((batched - got).abs().max())
    same_argmax = int(single.argmax()) == int(got.argmax())

    for req in requests:
        release(engine, req)

    # Against a decode at the same width the verify pass should be essentially
    # exact; against width 1 it may differ by as much as the shape itself does.
    ok = same_argmax and vs_batched <= state_floor and vs_single <= state_floor
    print(f"   max|decode@1     - verify[0]| = {vs_single:.4f}")
    print(f"   max|decode@{k + 1}     - verify[0]| = {vs_batched:.4f}   "
          f"(same batch width as verify)")
    print(f"   max|decode@1     - decode@{k + 1}| = {state_floor:.4f}   <- floor at this state")
    print(f"   argmax agrees = {same_argmax}")
    print(f"   gate 2 {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# gate 3 — teacher-forced fidelity
# ---------------------------------------------------------------------------

def gate_contract_at_anchors(engine: Engine, prompt_tokens: list[int], reference: list[int],
                             k: int, floor: float, label: str) -> tuple[int, int, int]:
    """At each anchor, one verify pass must equal K+1 sequential decode steps.

    Returns (rows_checked, hard_failures, tolerated_ties).
    """
    vocab_size = engine.config.hf_config.vocab_size
    # Spread anchors along the reference: short context, mid, and past the
    # 64-token block boundary so a second block is in play.
    anchors = [g for g in (1, 4, 12, 24, 36, 48, 60) if g < len(reference)]

    checked = hard = ties = 0

    for script in ("oracle", "zero", "partial", "mixed"):
        script_checked = script_hard = script_ties = 0

        for anchor_index, g in enumerate(anchors):
            prefix = prompt_tokens + reference[:g]

            # Ground truth for this anchor: K+1 decode steps from a fresh
            # prefill. Both this and the verify request below start from an
            # identical, prefill-built cache. K+1 steps, not K, because the
            # verify pass returns K+1 rows and row K is the bonus position.
            truth, truth_rows, gaps = sequential_steps(engine, prefix, k + 1)

            if script == "oracle":
                num_correct = k
            elif script == "zero":
                num_correct = 0
            elif script == "partial":
                num_correct = max(1, k // 2)
            else:
                num_correct = anchor_index % (k + 1)

            # `truth[i]` is the token following the prefill token, so it is
            # exactly what proposal i should be to get accepted.
            proposal = scripted_proposal(truth, num_correct, k, vocab_size)

            req, first = prefill_one(engine, prefix)
            engine.scheduler.block_manager.allocate_block_for_decode(req, extra_tokens=k)
            logits = engine.executor.verify([req], [proposal])[0]
            argmax = logits.argmax(dim=-1).tolist()
            release(engine, req)

            # Only rows whose whole prefix is the true continuation are
            # predictable; rows after the first corrupted proposal follow a
            # sequence that never existed.
            for j in range(num_correct + 1):
                script_checked += 1
                if argmax[j] == truth[j]:
                    continue
                if gaps[j] <= floor:
                    script_ties += 1
                else:
                    script_hard += 1
                    deviation = float((truth_rows[j] - logits[j].float()).abs().max())
                    print(f"     MISMATCH {label} {script} anchor g={g} row {j}: "
                          f"got {argmax[j]} want {truth[j]} "
                          f"(top-2 gap {gaps[j]:.3f} > floor {floor:.3f}, "
                          f"max|dlogit| {deviation:.3f})")

        checked += script_checked
        hard += script_hard
        ties += script_ties
        print(f"     {script:8s} K={k}: {len(anchors)} anchors, {script_checked:3d} rows checked, "
              + ("clean" if script_hard == 0 else f"{script_hard} HARD FAILURES")
              + (f", {script_ties} near-tie" if script_ties else ""))

    return checked, hard, ties


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="proposals per round")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    prompts = chat_prompts(tokenizer)

    engine = Engine(build_config())
    eos_ids = set(engine.config.eos_token_ids or set())

    floor = batch_shape_noise_floor(engine, prompts[0], args.k + 1, verbose=True)
    ok1 = gate_context_exact(engine, prompts[0])
    ok2 = gate_no_added_error(engine, prompts[0], args.k)

    print("\n== gate 3: one verify pass == K+1 sequential decode steps ==")
    checked_total, hard_total, tie_total = 0, 0, 0
    for i, prompt_tokens in enumerate(prompts):
        reference = reference_walk(engine, prompt_tokens, args.max_tokens, eos_ids)
        print(f"\n   prompt {i}: {tokenizer.decode(prompt_tokens)[-42:]!r} "
              f"-> {len(reference)} reference tokens")
        checked, hard, ties = gate_contract_at_anchors(engine, prompt_tokens, reference,
                                                       args.k, floor, f"p{i}")
        checked_total += checked
        hard_total += hard
        tie_total += ties

    ok3 = hard_total == 0
    print(f"\n   {checked_total} rows checked, {hard_total} hard mismatches, "
          f"{tie_total} near-tie reorderings tolerated (top-2 gap below the "
          f"{floor:.3f} noise floor)")
    print(f"   gate 3 {'PASSED' if ok3 else f'FAILED ({hard_total} hard mismatches)'}")

    all_ok = ok1 and ok2 and ok3
    print("\nGATE:", "PASS - verify pass == sequential decode"
          if all_ok else "FAIL - see above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
