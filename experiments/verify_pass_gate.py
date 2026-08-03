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

`control_batch_shape_noise` measures that floor, and the gates below are stated
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

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minivllm.config.config import Config  # noqa: E402
from minivllm.config.sampling import SamplingParams  # noqa: E402
from minivllm.engine.engine import Engine  # noqa: E402
from minivllm.engine.request import Request  # noqa: E402
from minivllm.scheduler.batch import Batch  # noqa: E402

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")

TARGET = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

PROMPTS = [
    "What is the meaning of life?",
    "How do I get started with LLMs?",
]


def build_config() -> Config:
    """The 4 GB RTX 3050 profile, as in experiments/engine_spec_gate.py.

    CUDA graphs stay off: they are a decode-path optimisation already shown
    equivalent to eager in Phase 3, and leaving them out keeps exactly one
    variable under test — one query row per request versus K+1.
    """
    return Config(
        model=TARGET,
        max_model_len=1024,
        max_num_batched_tokens=2048,
        max_num_batched_seqs=8,
        kv_cache_block_size=64,
        gpu_memory_utilization=0.9,
        use_cuda_graph=False,
    )


# ---------------------------------------------------------------------------
# driving the executor by hand
# ---------------------------------------------------------------------------

def prefill_one(engine: Engine, prompt_tokens: list[int]) -> tuple[Request, int]:
    """Submit one request and run its prefill, returning it and its first token.

    Afterwards `req.tokens` is `prompt + [first]` and the cache holds the
    prompt — the last committed token has not been fed through the model, which
    is the invariant every verify round assumes.
    """
    sp = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, max_tokens=1 << 30)
    req = Request(prompt_tokens, sp)
    engine.scheduler.submit(req)
    batch = engine.scheduler.schedule()
    assert batch is not None and batch.type == Batch.PREFILL and batch.requests == [req]
    tokens = engine.executor.execute(batch)
    # Bypass Scheduler.update: it also runs prefix-cache bookkeeping and
    # end-of-sequence handling, neither of which this harness wants.
    req.append_output_token(tokens[0])
    return req, tokens[0]


def release(engine: Engine, req: Request) -> None:
    if req in engine.scheduler.running:
        engine.scheduler.running.remove(req)
    if req.blocks:
        engine.scheduler.block_manager.deallocate(req)


def decode_logits(engine: Engine, requests: list[Request]) -> torch.Tensor:
    """One plain decode step for `requests`, eager, returning (batch, vocab)."""
    for req in requests:
        engine.scheduler.block_manager.allocate_block_for_decode(req)
    with torch.inference_mode():
        input_ids, ctx = engine.executor._build_decode_input(requests)
        return engine.executor.model(ctx, input_ids, ctx.positions).float()


# ---------------------------------------------------------------------------
# the control: how much does batch shape alone move the logits?
# ---------------------------------------------------------------------------

def control_batch_shape_noise(engine: Engine, prompt_tokens: list[int], width: int) -> float:
    """Decode one state at batch size 1 and at batch size `width`.

    No verify pass anywhere. Whatever this returns is the floor below which
    "different logits" means "different GEMM tiling", not "different maths".
    """
    print(f"\n== control: plain decode at batch 1 vs batch {width} (no verify pass) ==")
    requests = [prefill_one(engine, prompt_tokens)[0] for _ in range(width)]

    single = decode_logits(engine, requests[:1])[0]
    batched = decode_logits(engine, requests)

    floor = max(float((single - batched[i]).abs().max()) for i in range(width))
    spread = max(float((batched[0] - batched[i]).abs().max()) for i in range(1, width))
    mean_dev = float((single - batched[0]).abs().mean())

    for req in requests:
        release(engine, req)

    print(f"   max|batch1 - batch{width}| = {floor:.4f}   mean = {mean_dev:.4f}")
    print(f"   spread among identical rows within one batch = {spread:.4f} "
          f"(deterministic within a shape)")
    print(f"   => noise floor for this model/dtype: {floor:.4f}")
    return floor


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

def reference_walk(engine: Engine, prompt_tokens: list[int], max_tokens: int,
                   eos_ids: set[int]) -> list[int]:
    """Plain greedy decoding — the token sequence anchors are taken from."""
    req, first = prefill_one(engine, prompt_tokens)
    tokens = [first]

    while len(tokens) < max_tokens and tokens[-1] not in eos_ids:
        nxt = int(decode_logits(engine, [req])[0].argmax())
        tokens.append(nxt)
        req.append_output_token(nxt)

    release(engine, req)
    return tokens


def sequential_steps(engine: Engine, prefix: list[int], steps: int
                     ) -> tuple[list[int], list[torch.Tensor], list[float]]:
    """Prefill `prefix`, then take `steps` plain decode steps.

    Returns the tokens produced (the first from prefill, the rest from decode),
    their logit rows, and each step's top-2 gap. This is the behaviour the
    verify pass has to reproduce.
    """
    req, first = prefill_one(engine, prefix)
    with torch.inference_mode():
        tokens = [first]
        rows: list[torch.Tensor] = []
        gaps: list[float] = []

        for _ in range(steps):
            logits = decode_logits(engine, [req])[0]
            top2 = torch.topk(logits, 2).values
            rows.append(logits.clone())
            gaps.append(float(top2[0] - top2[1]))
            nxt = int(logits.argmax())
            tokens.append(nxt)
            req.append_output_token(nxt)

    release(engine, req)
    # `tokens[0]` came from the prefill; rows[i] is the step that produced
    # tokens[i + 1].
    return tokens[1:], rows, gaps


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
            proposal = [t if i < num_correct else (t + 1) % vocab_size
                        for i, t in enumerate(truth[:k])]

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
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)
        for p in PROMPTS
    ]

    engine = Engine(build_config())
    eos_ids = set(engine.config.eos_token_ids or set())

    floor = control_batch_shape_noise(engine, prompts[0], args.k + 1)
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
