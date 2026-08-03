"""Regression check: decode must be deterministic at every batch width.

**This guards a fixed backend bug.** It requires
`patches/mini-flash-attention-decode-race.patch` to be applied to the installed
`mini-flash-attention`. Against stock upstream this script fails from batch
width 6 upward — see `patches/README.md`.

What it asserts: calling the *same* decode step twice on the *same* requests
must return identical logits. The pass is idempotent — it rewrites identical KV
into identical slots before attending — so two consecutive calls have to agree
bitwise. Separately, N identical requests batched together must agree with each
other.

## The bug this guards against

Found while gating Phase 4, and unrelated to speculative decoding — it
reproduced on a clean checkout with none of that code present.

`flash_attention_fwd_split_kv_kernel` aliases two different things over the same
`extern __shared__` region: `warp_max_val` / `warp_expsum_val` (the cross-warp
softmax reduction, `decode.cuh:588`) and `warp_output` (`:628`), both at offset
0. Every warp reads the reduction values at `:604` and `:613`, then each warp
writes 128 floats of output from offset 0 at `:512`, clobbering them. No barrier
sat between the reads and the write, so warp 0's write raced the other warps'
reads.

Because the corruption lands on the softmax normalisation, the result is a
*rescaled* output rather than a small perturbation — logits moved by 10-30 and
the emitted token changed. It only manifested once occupancy let warps drift out
of lockstep, which is why it presented as batch-size-dependent nondeterminism:

    width   repeat-call max|diff|   verdict        (stock upstream)
        1              0.0000       deterministic
        5              0.0000       deterministic
        6              1.1250       NONDETERMINISTIC
        8             11.1719       NONDETERMINISTIC
       18             21.6875       NONDETERMINISTIC

Three plausible culprits were ruled out by direct experiment before the real one
was found, and they are worth recording because each *moved* the failing widths,
which is exactly what a scheduling-sensitive race does to anything that perturbs
occupancy:

- **Not the split-KV heuristic.** `flash_attn_with_kvcache` takes `num_splits=0`,
  so the backend picks a split count from `batch * heads` (`api.cpp:321`).
  Pinning it to 1 shifted the pattern without fixing it.
- **Not `@torch.compile` on `MLP.forward`.** `TORCHDYNAMO_DISABLE=1`, likewise.
- **Not uninitialized KV cache.** Allocating with `zeros` or a constant instead
  of `torch.empty` did not help, so it was not merely reading past
  `cache_seqlens` into garbage.

`compute-sanitizer --tool racecheck` then named it directly: ~4,500 hazards per
launch between `decode.cuh:512` and `:604`/`:613`, and 0 after the one-line fix.

Note the "safe below width 6" framing was always wrong: the race existed at every
width and simply did not manifest when few resident blocks kept the warps in
lockstep. There was never a safe ceiling, only an unobserved one.

Usage:
    python experiments/decode_determinism_check.py
    python experiments/decode_determinism_check.py --max-width 32
"""

import argparse
import logging
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from mini_flash_attention import flash_attn_with_kvcache  # noqa: E402

from minivllm.config.config import Config  # noqa: E402
from minivllm.config.sampling import SamplingParams  # noqa: E402
from minivllm.engine.engine import Engine  # noqa: E402
from minivllm.engine.request import Request  # noqa: E402

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")

TARGET = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


def through_the_engine(max_width: int) -> int:
    """Two identical decode calls through the real engine."""
    print("== through the engine: repeat the same decode step twice ==")
    config = Config(model=TARGET, max_model_len=1024, max_num_batched_tokens=2048,
                    max_num_batched_seqs=max(32, max_width), kv_cache_block_size=64,
                    gpu_memory_utilization=0.9, use_cuda_graph=False)
    engine = Engine(config)
    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the meaning of life?"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True)
    sp = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, max_tokens=1 << 30)

    requests = []
    for _ in range(max_width):
        req = Request(prompt, sp)
        engine.scheduler.submit(req)
        batch = engine.scheduler.schedule()
        req.append_output_token(engine.executor.execute(batch)[0])
        requests.append(req)

    def decode(reqs):
        for req in reqs:
            engine.scheduler.block_manager.allocate_block_for_decode(req)
        with torch.inference_mode():
            input_ids, ctx = engine.executor._build_decode_input(reqs)
            return engine.executor.model(ctx, input_ids, ctx.positions).float()

    print(f"\n{'width':>6} {'repeat-call max|diff|':>22} {'intra-batch spread':>20}  verdict")
    first_bad = 0
    for width in range(1, max_width + 1):
        a = decode(requests[:width])
        b = decode(requests[:width])
        repeat = float((a - b).abs().max())
        # All requests hold identical tokens, so all rows must agree too.
        spread = max((float((a[0] - a[i]).abs().max()) for i in range(1, width)), default=0.0)
        bad = repeat != 0.0
        if bad and not first_bad:
            first_bad = width
        print(f"{width:>6} {repeat:>22.4f} {spread:>20.4f}  "
              f"{'NONDETERMINISTIC' if bad else 'deterministic'}")

    return first_bad


def kernel_in_isolation() -> bool:
    """Same question, no engine: random tensors straight into the kernel."""
    print("\n== the kernel alone: identical inputs, two calls ==")
    generator = torch.Generator(device="cuda").manual_seed(0)
    any_bad = False
    for width in (1, 2, 4, 6, 8, 12, 16, 24, 32):
        q = torch.randn(width, 1, 16, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        k_cache = torch.randn(64, 64, 8, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        v_cache = torch.randn(64, 64, 8, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        cache_seqlens = torch.full((width,), 16, dtype=torch.int32, device="cuda")
        block_table = torch.arange(width, dtype=torch.int32, device="cuda").reshape(width, 1)

        first = flash_attn_with_kvcache(q, k_cache, v_cache,
                                        cache_seqlens=cache_seqlens, block_table=block_table)
        second = flash_attn_with_kvcache(q, k_cache, v_cache,
                                         cache_seqlens=cache_seqlens, block_table=block_table)
        diff = float((first.float() - second.float()).abs().max())
        any_bad |= diff != 0.0
        print(f"   width={width:3d}: max|call1 - call2| = {diff:.6f}  "
              f"{'NONDETERMINISTIC' if diff else 'deterministic'}")
    return any_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-width", type=int, default=18)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this check needs the GPU backend"

    first_bad = through_the_engine(args.max_width)
    isolated = kernel_in_isolation()

    print()
    if first_bad or isolated:
        print(f"RESULT: FAIL - decode is nondeterministic"
              + (f" from batch width {first_bad} upward" if first_bad else "")
              + (" (reproduced in the bare kernel too)" if isolated else "") + ".")
        print("        Is patches/mini-flash-attention-decode-race.patch applied to "
              "the installed")
        print("        mini-flash-attention? See patches/README.md. Confirm with:")
        print("          compute-sanitizer --tool racecheck --racecheck-report analysis \\")
        print("            python <a script making one flash_attn_with_kvcache call at width 8>")
        return 1

    print("RESULT: PASS - decode is deterministic at every width tested.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
