"""Classify a speculative-decoding output divergence: tie, drift, or real bug?

`benchmark/spec_sweep.py` compares each configuration's tokens against plain
decoding's. That check is worth having only if there is something to do when it
fires, and "byte-identical greedy output is a tolerance, not a theorem" is not an
answer on its own — it is a hypothesis, and this script tests it.

There are **three** things a divergence can be, and they need different evidence:

1. **A near-tie the batch shape reordered.** Speculative decoding is exactly
   distribution-preserving, so under greedy sampling it emits the target's own
   argmax — but that argmax is taken over logits that shift with the shape the
   model ran at. Where the target's own top-2 gap is below that shift, both
   orderings are defensible and the emitted token can flip harmlessly.
2. **KV drift.** Past a measured batch width, a verify pass writes *different
   K/V into the cache* than a decode step would, permanently. The two runs then
   decode over different numbers and can part company at a step where plain
   decode was perfectly certain. `experiments/kv_shape_drift.py` measures where
   this begins; above it byte identity is unachievable at any acceptance rate.
3. **A bug.** Everything else.

**The trap.** The top-2 gap test only distinguishes (1) from (3), and only while
both runs share a cache. Above the drift width it answers a question nobody
asked — "would plain decode have been sure?" — and confidently reports a bug. So
the width is checked first and the gap is reported without a verdict when it does
not apply.

**A second trap, in how the gap is measured.** Re-deriving the step by prefilling
`prompt + tokens[:at]` and taking one decode step gives a *third* cache state
that neither run was ever in, because a prefill and a decode walk write different
K/V for the same tokens. The gap has to come from walking plain decode from the
prompt. An earlier version of this script got that wrong.

Run `python experiments/decode_determinism_check.py` before believing anything
here: a missing backend patch makes decode itself nondeterministic above batch
width 6, which would produce divergences unrelated to any of the three.

Usage:
    python experiments/spec_divergence.py --k 8
    python experiments/spec_divergence.py --k 8 --method draft
    python experiments/spec_divergence.py --k 8 --workload repetitive
"""

import argparse
import os
import sys

# Must precede the torch import: inductor reads this at config-module import.
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from benchmark.spec_sweep import KV_IDENTICAL_MAX_WIDTH  # noqa: E402
from minivllm.config.sampling import SamplingParams  # noqa: E402
from tests.spec_harness import (  # noqa: E402
    DRAFT, PROMPTS, REPETITIVE_PROMPTS, TARGET,
    batch_shape_noise_floor, build_engine, chat_prompts, free_gpu_memory,
    sequential_steps,
)


def generate(spec: bool, method: str, k: int, prompts, max_tokens: int, draft: str,
             cuda_graph: bool = True):
    engine = build_engine(spec=spec, draft=draft, k=k, cuda_graph=cuda_graph, method=method)
    outputs = engine.generate(
        prompts,
        SamplingParams(temperature=1.0, top_k=0, top_p=1.0, max_tokens=max_tokens),
        use_tqdm=False)
    tokens = [o["tokens"] for o in outputs]
    del engine
    free_gpu_memory()
    return tokens


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--method", choices=("ngram", "draft"), default="ngram")
    ap.add_argument("--workload", choices=("open", "repetitive"), default="open")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--eager", action="store_true",
                    help="run both sides with CUDA graphs off. If a divergence "
                         "disappears here it is about graph capture or replay "
                         "padding, not about the round")
    ap.add_argument("--requests", type=int, default=None,
                    help="use only the first N prompts. Batch width changes the "
                         "verify pass's shape, so this bisects shape-dependent "
                         "divergences")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this diagnostic needs the GPU"

    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    if args.workload == "repetitive":
        prompts = chat_prompts(tokenizer, REPETITIVE_PROMPTS, enable_thinking=False)
    else:
        prompts = chat_prompts(tokenizer, PROMPTS, enable_thinking=True)

    if args.requests:
        prompts = prompts[:args.requests]
    draft = TARGET if args.method == "draft" else DRAFT
    graphs = not args.eager

    # A round puts K+1 query rows in the batch dimension for every request, and
    # past a measured width that GEMM writes different KV than decode's does.
    width = len(prompts) * (args.k + 1)

    print(f"\n== {args.method} K={args.k} vs plain decode, {args.workload} workload, "
          f"{len(prompts)} request(s), graphs {'on' if graphs else 'off'} ==")
    print(f"   verify pass runs at batch width {width}; KV stays bitwise identical to "
          f"decode's up to width {KV_IDENTICAL_MAX_WIDTH}")
    baseline = generate(False, "draft", args.k, prompts, args.max_tokens, DRAFT, graphs)
    speculative = generate(True, args.method, args.k, prompts, args.max_tokens, draft, graphs)

    engine = build_engine(cuda_graph=False)
    floor = batch_shape_noise_floor(engine, prompts[0], args.k + 1)
    print(f"\n   batch-shape noise floor for this model/dtype: {floor:.4f}")
    print(f"   (plain decode at batch 1 vs batch {args.k + 1}, no verify pass anywhere)")

    verdicts = []
    for i, (base, spec) in enumerate(zip(baseline, speculative)):
        at = first_divergence(base, spec)
        if at is None:
            print(f"\n   request {i}: identical, {len(base)} tokens")
            verdicts.append("identical")
            continue

        print(f"\n   request {i}: diverges at token {at} of {len(base)}")
        print(f"     plain decode wanted {base[at]} "
              f"({tokenizer.decode([base[at]])!r})")
        print(f"     speculation emitted {spec[at]} "
              f"({tokenizer.decode([spec[at]])!r})")

        # Re-derive that step by *walking* plain decode from the prompt, not by
        # prefilling `prompt + base[:at]` and taking one step.
        #
        # This distinction is the whole diagnostic. A prefill and a decode walk
        # write subtly different K/V for the same tokens — bf16 accumulation
        # differs with the shape the projection ran at — so a freshly prefilled
        # prefix is a *third* cache state that neither run was ever in. Measuring
        # the gap there answers "would a fresh prefill have been sure?", which is
        # not the question. The question is how sure the plain-decode run was, in
        # the state plain decode was actually in.
        #
        # An earlier version of this script got that wrong and reported a 14.75
        # gap — a confident "hard mismatch" — at a step where the run itself was
        # nearly tied.
        _tokens, _rows, gaps = sequential_steps(engine, prompts[i], at + 1)
        gap = gaps[at]

        near_tie = gap <= floor
        print(f"     target's own top-2 gap at that step: {gap:.4f}")

        if width > KV_IDENTICAL_MAX_WIDTH:
            # The gap test does not apply here and reporting it as a verdict
            # would be wrong. It asks "was plain decode unsure?", which is only
            # the right question when both runs decoded over the same cache. A
            # round this wide writes different K/V than a decode step does, so
            # the two runs hold different numbers for the same tokens and can
            # part company at a step where plain decode was perfectly sure.
            verdicts.append("kv-drift")
            print(f"     -> KV DRIFT: verify width {width} exceeds "
                  f"{KV_IDENTICAL_MAX_WIDTH}, so the two runs are not decoding "
                  f"over the same cache. The gap above is not evidence either way.")
        else:
            verdicts.append("near-tie" if near_tie else "HARD")
            print(f"     -> {'NEAR TIE' if near_tie else 'HARD MISMATCH'}: "
                  f"gap is {'below' if near_tie else 'ABOVE'} the {floor:.4f} floor")

        head = 6
        print(f"     continuation, plain:       "
              f"{tokenizer.decode(base[at:at + head])!r}")
        print(f"     continuation, speculative: "
              f"{tokenizer.decode(spec[at:at + head])!r}")

    del engine
    free_gpu_memory()

    hard = [i for i, v in enumerate(verdicts) if v == "HARD"]
    drift = [i for i, v in enumerate(verdicts) if v == "kv-drift"]
    print()
    if hard:
        print(f"VERDICT: HARD MISMATCH on request(s) {hard}. This is not a tie being "
              f"reordered and the width is low enough that the caches should have "
              f"matched — investigate. Confirm the backend patch first with "
              f"experiments/decode_determinism_check.py.")
        return 1
    if drift:
        print(f"VERDICT: KV drift on request(s) {drift}. At verify width {width} the "
              f"round writes different K/V than plain decode does "
              f"(experiments/kv_shape_drift.py), so byte-identical output is not "
              f"achievable here at any acceptance rate. Lower K, or fewer concurrent "
              f"requests, to get back under width {KV_IDENTICAL_MAX_WIDTH}.")
        return 0
    if all(v == "identical" for v in verdicts):
        print("VERDICT: identical. Nothing to classify.")
    else:
        print("VERDICT: every divergence is a near-tie the batch shape reordered. "
              "Expected behaviour, not a bug — see the tolerance note in PROGRESS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
