"""At what batch width does a verify pass stop writing the same KV as decode?

This is the measurement that explains why byte-identical greedy output survives
K=4 and dies at K=8, and it is a sharper statement than the "batch shape moves
logits by up to 0.5" note the rest of the project works from.

**The distinction that matters.** Logit noise is *transient*: the model runs at a
new shape, the logits come out slightly different, an argmax may flip at a genuine
tie, and nothing persists. KV noise is *permanent*. A verify pass writes K and V
into the paged cache for every token it processes, and those bytes stay there for
the rest of the sequence. If the projection GEMM retiles at the verify pass's
width and produces different K/V than a decode step would have, the two runs no
longer share a cache — and they can then diverge at a step that is not remotely a
tie, arbitrarily far downstream.

That is exactly what happens here. Measured on the dev host:

    rows/request   batch width   max|dK| vs decode   max|dV|
             1              2         0.000000       0.000000
             5             10         0.000000       0.000000
             9             18         1.000000       1.125000

So a K=4 round over 2 requests (width 10) writes *bitwise identical* KV to a
plain decode step, and a K=8 round over 2 requests (width 18) does not. Greedy
output is byte-identical in the first case and provably cannot be guaranteed in
the second — not because anything is wrong, but because the two runs are no
longer computing over the same numbers.

**What this is not.** It is not the decode-kernel race (that is fixed and
`experiments/decode_determinism_check.py` gates it; this is deterministic, it
just differs by shape). It is not CUDA graphs (reproduces eager). It is not the
block reservation (a K=8 config that never actually speculates is identical). It
is cuBLAS choosing a different tiling for a different M, which is expected
behaviour from the library and not something this engine can or should fix.

**What to do with it.** Treat the width where dK first becomes nonzero as the
ceiling on byte-identical output, and state correctness claims above it against a
measured tolerance rather than against equality. `benchmark/spec_sweep.py`
reports it per configuration for exactly this reason.

Usage:
    python experiments/kv_shape_drift.py
    python experiments/kv_shape_drift.py --rows 1 2 3 4 5 6 7 8 9 --requests 2
"""

import argparse
import os
import sys

# Must precede the torch import: inductor reads this at config-module import.
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from tests.spec_harness import (  # noqa: E402
    TARGET, build_engine, chat_prompts, free_gpu_memory, prefill_one, release,
)


def measure(engine, prompt, rows_per_request: int, num_requests: int, extra: int):
    """Write KV for one token at a given batch width; return it.

    Row 0 of each request is the request's last committed token at its real
    position — the row a plain decode step would have run. The remaining rows
    are filler at later positions, present only to make the batch the width a
    verify pass would have had. Their content is irrelevant: what is being
    measured is whether row 0's *own* K/V changes because of who it shares a
    GEMM with.
    """
    executor = engine.executor
    block_manager = engine.scheduler.block_manager
    block_size = engine.config.kv_cache_block_size

    requests = [prefill_one(engine, prompt)[0] for _ in range(num_requests)]
    for req in requests:
        block_manager.allocate_block_for_decode(req, extra_tokens=extra)

    pos = len(requests[0].tokens) - 1
    with torch.inference_mode():
        if rows_per_request == 1:
            input_ids, ctx = executor._build_decode_input(requests)
        else:
            rows = [[(req.tokens[-1], pos)]
                    + [(0, pos + 1 + j) for j in range(rows_per_request - 1)]
                    for req in requests]
            input_ids, ctx = executor._build_paged_rows(requests, rows)
        executor.model(ctx, input_ids, ctx.positions)

    slot = requests[0].blocks[pos // block_size] * block_size + pos % block_size
    block, offset = divmod(slot, block_size)
    k = executor.kv_cache[0, :, block, offset].clone()
    v = executor.kv_cache[1, :, block, offset].clone()

    for req in requests:
        release(engine, req)
    return k, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 5, 7, 9, 17],
                    help="query rows per request. A K-token round has K+1.")
    ap.add_argument("--requests", type=int, default=2)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this measurement needs the GPU"

    engine = build_engine(cuda_graph=False)
    prompt = chat_prompts(AutoTokenizer.from_pretrained(TARGET))[0]

    print(f"\n== KV written for one token, at the batch widths a round produces ==")
    print(f"   {args.requests} request(s); row 0 is the token a decode step would write\n")
    print(f"   {'rows/req':>9} {'K':>4} {'width':>6} {'max|dK|':>12} {'max|dV|':>12}   verdict")
    print("   " + "-" * 62)

    reference = None
    first_drift = None
    for rows in sorted(args.rows):
        # A round's slack is K = rows - 1 tokens; reserve it so the filler rows
        # have blocks to land in.
        k, v = measure(engine, prompt, rows, args.requests, extra=max(rows - 1, 0))
        if reference is None:
            reference = (k, v)
        dk = float((k - reference[0]).abs().max())
        dv = float((v - reference[1]).abs().max())
        drifted = dk > 0 or dv > 0
        if drifted and first_drift is None:
            first_drift = rows * args.requests
        print(f"   {rows:>9} {rows - 1:>4} {rows * args.requests:>6} "
              f"{dk:>12.6f} {dv:>12.6f}   "
              f"{'DIFFERS from decode' if drifted else 'bitwise identical'}")

    del engine
    free_gpu_memory()

    print()
    if first_drift is None:
        print("RESULT: every width tested writes the same KV as a plain decode step, so "
              "byte-identical greedy output is achievable across all of them.")
    else:
        print(f"RESULT: KV first differs at batch width {first_drift}. At and above it, "
              f"a speculative round leaves the cache holding different numbers than plain "
              f"decode would have, so byte-identical output is not achievable at any "
              f"acceptance rate and a divergence there is not a bug. Below it, byte "
              f"identity is a reasonable gate and this project uses it as one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
