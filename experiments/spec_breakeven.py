"""How good does the draft have to be before speculation pays for itself?

Phases 2-6 established that speculative decoding here is *correct*. This script
asks the question that decides whether it is *worth it*, and it deliberately
runs before Phase 1 (training a draft), because the answer changes what Phase 1
should aim at — or whether it is worth doing on this hardware at all.

A round costs one `propose()` (K sequential draft passes) plus one `verify()`
(one target pass at K+1 rows per request) plus the rejection sampler, and
returns `M+1` tokens where M is how many proposals survived. Plain decoding
spends one step per token. So speculation wins exactly when

    E[M] + 1  >  T_round / T_decode

and the break-even acceptance is whatever makes that an equality. Everything
here measures the right-hand side.

**This needs no trained draft.** Round cost is set by the draft's *architecture*
— layer count, hidden size, and how much of the time is kernel-launch overhead —
not by its weights. The random-init draft already in the repo has exactly the
right shape, which is what makes this an afternoon's measurement rather than
something that has to wait for Phase 1.

**Converting E[M] to an acceptance rate.** Treating per-token acceptance as iid
with probability `a`, a round survives to step i with probability `a**i`, so
`E[M] = sum(a**i for i in 1..K)`. That is an idealisation — real acceptance is
correlated, since a draft that is wrong once is usually wrong because it lost
the thread — but it is the standard way to state the number and it is monotonic
in `a`, so the inversion is well defined.

**Two measurement traps, both hit for real in this project.**

1. *Do not compare across engines built in one process.* Two `Engine`s built
   back to back report wildly different throughput for identical work, because
   the second inherits warm `torch.compile` artefacts and cuBLAS autotuning.
   Every number in the main table below therefore comes from **one** engine,
   with each timed path warmed on its own before it is timed.
2. *The production baseline uses CUDA graphs and the speculative path cannot.*
   Graphs are captured per decode batch size for one query row per request; a
   verify pass has K+1. Comparing eager-vs-eager would quietly flatter
   speculation, so the graph-accelerated decode step is measured too and the
   break-even is restated against it. That second number is the honest one for
   "should I turn this on".

**Cross-check the microbenchmark before believing it.** `--end-to-end` times
`Engine.generate` instead, one configuration per process, and its numbers must
agree with the table. They did when this was written:

    mode          per-step, end to end     per-step, microbenchmark
    base_eager              76.1 ms                    68.6 ms
    base_graph               9.4 ms                     8.7 ms
    spec (K=4)             146.0 ms                   134.9 ms

The end-to-end column runs a little high because it also carries scheduling and
sampling per step, and it decodes two requests where the microbenchmark decodes
one. The point is that the two agree on the *ratios*, which is what break-even
depends on: eager decode is ~8x a graphed one, and a K=4 round is ~15x.

Run it as three separate invocations — never in one process, per trap 1 above:

    for m in base_eager base_graph spec; do
        python experiments/spec_breakeven.py --end-to-end $m
    done

Usage:
    python experiments/spec_breakeven.py
    python experiments/spec_breakeven.py --k 1 2 4 8 --batch 1 --context 256
    python experiments/spec_breakeven.py --batch 4 --iters 50
    python experiments/spec_breakeven.py --end-to-end spec
"""

import argparse
import logging
import os
import statistics
import sys
import time

# Must precede the torch import: inductor reads this at config-module import.
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minivllm.config.sampling import SamplingParams  # noqa: E402
from minivllm.engine.engine import Engine  # noqa: E402
from minivllm.engine.request import Request  # noqa: E402
from minivllm.scheduler.batch import Batch  # noqa: E402

from tests.spec_harness import (  # noqa: E402
    GREEDY,
    TARGET,
    build_engine,
    chat_prompts,
    free_gpu_memory,
)

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def timed(fn, warmup: int, iters: int) -> dict:
    """Median wall time of `fn`, in milliseconds.

    Every timed callable here is idempotent — it rewrites the same KV into the
    same slots and never commits a token — so repeating it measures the same
    work each time rather than walking the sequence forward.

    Median rather than mean: on a laptop GPU sharing a display, a handful of
    samples land far above the rest and the mean chases them. The minimum is
    reported too, as the cleanest view of the work itself.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)

    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "mean": statistics.fmean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# break-even algebra
# ---------------------------------------------------------------------------

def expected_accepted(acceptance: float, k: int) -> float:
    """E[M] for iid per-token acceptance probability `acceptance`."""
    return sum(acceptance ** i for i in range(1, k + 1))


def required_acceptance(needed_tokens: float, k: int) -> float | None:
    """Invert `expected_accepted`. None when even a perfect draft is not enough."""
    if needed_tokens <= 0.0:
        return 0.0
    if needed_tokens > k:
        return None
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2
        if expected_accepted(mid, k) < needed_tokens:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# ---------------------------------------------------------------------------
# putting requests into a decode-ready state
# ---------------------------------------------------------------------------

def make_requests(engine: Engine, prompt_tokens: list[int], batch: int,
                  context: int) -> list[Request]:
    """Prefill `batch` requests, each holding `context` tokens of KV.

    The prompt is cycled to reach the requested length. Token identity is
    irrelevant to timing — none of these kernels branch on values — and using a
    real greedy walk instead would add a minute of setup to measure the same
    thing.

    Afterwards each request is in the state a decode step assumes: the cache
    holds `req.tokens[:-1]` and the last committed token is unprocessed.
    """
    assert context > len(prompt_tokens), (
        f"context {context} must exceed the prompt's {len(prompt_tokens)} tokens"
    )
    filler = (prompt_tokens * (context // len(prompt_tokens) + 1))[:context - 1]

    requests = []
    for _ in range(batch):
        req = Request(filler, GREEDY)
        engine.scheduler.submit(req)
        batch_obj = engine.scheduler.schedule()
        assert batch_obj is not None and batch_obj.type == Batch.PREFILL
        token = engine.executor.execute(batch_obj)[batch_obj.requests.index(req)]
        req.append_output_token(token)
        requests.append(req)

    assert all(len(r.tokens) == context for r in requests)
    return requests


# ---------------------------------------------------------------------------
# the measurements
# ---------------------------------------------------------------------------

def measure(engine: Engine, requests: list[Request], k_values: list[int],
            warmup: int, iters: int) -> tuple[dict, dict]:
    """Time a plain decode step, then a full round at each K, in one engine."""
    executor = engine.executor
    block_manager = engine.scheduler.block_manager

    def plain_decode():
        with torch.inference_mode():
            input_ids, ctx = executor._build_decode_input(requests)
            executor.forward(ctx, input_ids)

    decode = timed(plain_decode, warmup, iters)

    rounds = {}
    for k in sorted(k_values):
        # propose() reads K off the config at call time, so one engine can sweep
        # it. The blocks are reserved by hand here because the scheduler is not
        # driving these steps.
        engine.config.num_speculative_tokens = k
        for req in requests:
            block_manager.allocate_block_for_decode(req, extra_tokens=k)

        batch = Batch(Batch.DECODE, requests)
        proposals, q = executor.propose(requests)
        p = executor.sampler.greedy_probs(executor.verify(requests, proposals))
        draft_tokens = torch.tensor(proposals, dtype=torch.int64, device=q.device)

        rounds[k] = {
            "round": timed(lambda: executor.execute_speculative(batch), warmup, iters),
            "propose": timed(lambda: executor.propose(requests), warmup, iters),
            "verify": timed(lambda: executor.verify(requests, proposals), warmup, iters),
            "sample": timed(
                lambda: executor.sampler.rejection_sample(draft_tokens, q, p), warmup, iters),
        }

    return decode, rounds


def report(decode: dict, rounds: dict, batch: int, context: int,
           graph_decode: dict | None) -> None:
    print(f"\n{'=' * 78}")
    print(f"Break-even acceptance   batch={batch}  context={context} tokens")
    print(f"{'=' * 78}")

    t1 = decode["median"]
    print(f"\nplain decode step (eager): {t1:7.2f} ms median  "
          f"({decode['min']:.2f} min, sd {decode['stdev']:.2f})")
    if graph_decode is not None:
        tg = graph_decode["median"]
        print(f"plain decode step (CUDA graph): {tg:7.2f} ms median  "
              f"-> graphs are {t1 / tg:.2f}x faster than eager")

    print(f"\n{'K':>3} {'round':>9} {'propose':>9} {'verify':>9} {'sample':>8} "
          f"{'round/T1':>9} {'need E[M]':>10} {'need acc':>9} {'max x':>7}")
    print(f"{'-' * 78}")

    for k, timings in sorted(rounds.items()):
        tr = timings["round"]["median"]
        ratio = tr / t1
        needed = ratio - 1.0
        acceptance = required_acceptance(needed, k)
        best = (k + 1) * t1 / tr

        print(f"{k:>3} {tr:>8.2f}ms {timings['propose']['median']:>8.2f}ms "
              f"{timings['verify']['median']:>8.2f}ms {timings['sample']['median']:>7.2f}ms "
              f"{ratio:>9.2f} {needed:>10.2f} "
              f"{('impossible' if acceptance is None else f'{acceptance:>8.1%}'):>9} "
              f"{best:>6.2f}x")

    print("\n  round/T1   how many plain decode steps one round costs")
    print("  need E[M]  proposals a round must average just to break even")
    print("  need acc   the per-token acceptance rate that produces that E[M]")
    print("  max x      speedup at 100% acceptance — the ceiling for this K")

    if graph_decode is not None:
        tg = graph_decode["median"]
        print(f"\n{'-' * 78}")
        print("Against the CUDA-graph baseline, which is what production decode "
              "actually runs.\nSpeculative rounds cannot use graphs (they are "
              "captured for one query row per\nrequest; a round has K+1), so this "
              "is the comparison that decides whether to\nturn the feature on:")
        print(f"\n{'K':>3} {'round/T1_graph':>15} {'need E[M]':>10} {'need acc':>10} {'max x':>7}")
        print(f"{'-' * 78}")
        for k, timings in sorted(rounds.items()):
            ratio = timings["round"]["median"] / tg
            needed = ratio - 1.0
            acceptance = required_acceptance(needed, k)
            best = (k + 1) * tg / timings["round"]["median"]
            print(f"{k:>3} {ratio:>15.2f} {needed:>10.2f} "
                  f"{('impossible' if acceptance is None else f'{acceptance:.1%}'):>10} "
                  f"{best:>6.2f}x")


def end_to_end(mode: str, k: int, max_tokens: int) -> int:
    """Time `Engine.generate` for one configuration, to validate the table above.

    One configuration per process on purpose. Two engines built back to back in
    one process do not produce comparable numbers — the second inherits warm
    `torch.compile` artefacts and cuBLAS autotuning — and this measurement
    exists precisely to be trustworthy.
    """
    spec = mode == "spec"
    engine = build_engine(spec=spec, k=k, cuda_graph=(mode == "base_graph"))
    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    prompts = chat_prompts(tokenizer)

    # A short throwaway run first: the first generate pays for compilation and
    # autotuning, which is exactly the contamination this mode is guarding.
    engine.generate(prompts, SamplingParams(temperature=1.0, top_k=0, top_p=1.0,
                                            max_tokens=8), use_tqdm=False)

    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = engine.generate(
        prompts, SamplingParams(temperature=1.0, top_k=0, top_p=1.0,
                                max_tokens=max_tokens), use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

    tokens = sum(len(o["tokens"]) for o in outputs)
    stats = engine.metrics.stats()
    steps = engine.metrics.decode_steps

    print(f"\n{mode:12s} {tokens:4d} tokens in {wall:6.2f}s = {tokens / wall:7.1f} tok/s")
    print(f"             {steps} decode steps -> {engine.metrics.decode_time / steps * 1000:.1f} ms "
          f"per step")
    if spec:
        print(f"             acceptance {stats.acceptance_rate:.1%}, "
              f"{stats.tokens_per_request_step:.2f} tokens per request per step")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--batch", type=int, default=1,
                    help="requests decoded together (1 is the single-stream "
                         "latency case speculation targets)")
    ap.add_argument("--context", type=int, default=256,
                    help="tokens of KV each request holds while timing")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-cuda-graph-baseline", dest="graph_baseline",
                    action="store_false",
                    help="skip the second engine that measures graph-accelerated decode")
    ap.add_argument("--end-to-end", choices=("base_eager", "base_graph", "spec"),
                    default=None,
                    help="instead of the table, time Engine.generate for one "
                         "configuration; run once per configuration, in separate "
                         "processes, to validate the table")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="--end-to-end only: tokens to generate per prompt")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this measurement needs the GPU"

    if args.end_to_end:
        return end_to_end(args.end_to_end, max(args.k), args.max_tokens)

    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    prompt = chat_prompts(tokenizer)[0]

    # One engine for every number in the main table: propose, verify and plain
    # decode all measured against the same warm process state.
    engine = build_engine(spec=True, k=max(args.k), cuda_graph=False)
    requests = make_requests(engine, prompt, args.batch, args.context)
    decode, rounds = measure(engine, requests, args.k, args.warmup, args.iters)
    del engine, requests
    free_gpu_memory()

    graph_decode = None
    if args.graph_baseline:
        # A separate engine, because Phase 5 skips graph capture when
        # speculation is on — no captured graph could match a K+1-row verify
        # pass, so capturing them would only reserve a pool nothing draws from.
        engine = build_engine(spec=False, cuda_graph=True)
        requests = make_requests(engine, prompt, args.batch, args.context)
        executor = engine.executor

        def graph_decode_step():
            with torch.inference_mode():
                input_ids, ctx = executor._build_decode_input(requests)
                executor.forward(ctx, input_ids)

        assert executor.graph_runner is not None, "expected captured CUDA graphs"
        graph_decode = timed(graph_decode_step, args.warmup, args.iters)
        del engine, requests, executor
        free_gpu_memory()

    report(decode, rounds, args.batch, args.context, graph_decode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
