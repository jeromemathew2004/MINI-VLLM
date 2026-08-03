"""Decode is nondeterministic for batch widths >= 6. Reproduction.

**This is a pre-existing engine bug, not a speculative-decoding one.** It was
found while gating Phase 4 and reproduces on a clean checkout with no
speculative-decoding code involved. It is recorded here because it bounds what
any correctness gate in this repo can claim, and because it affects normal
serving far more than it affects speculative decoding.

What happens: calling the *same* decode step twice on the *same* requests
returns different logits. The decode pass is idempotent — it rewrites identical
KV into identical slots before attending — so two consecutive calls must agree
bitwise. Below batch width 6 they do. At and above it they diverge, by as much
as 20+ logits, and at some widths the argmax itself changes, which means the
engine emits a different token.

Observed on an RTX 3050 (sm_86), Qwen3-0.6B, bf16, block size 64:

    width   repeat-call max|diff|   verdict
        1              0.0000       deterministic
        2              0.0000       deterministic
        4              0.0000       deterministic
        5              0.0000       deterministic
        6              1.1250       NONDETERMINISTIC
        7              6.0234       NONDETERMINISTIC
        8             11.1719       NONDETERMINISTIC
       18             21.6875       NONDETERMINISTIC

Ruled out, each by direct experiment:

- **Not the split-KV heuristic.** `flash_attn_with_kvcache` takes `num_splits=0`,
  which makes the backend pick a split count from `batch * heads`
  (`csrc/mfa/api.cpp:321`). Pinning `num_splits=1` moves which widths break but
  does not fix it.
- **Not `@torch.compile` on `MLP.forward`.** `TORCHDYNAMO_DISABLE=1` likewise
  only shifts the pattern.
- **Not uninitialized KV cache.** Allocating the cache with `zeros` or a large
  constant instead of `torch.empty` does not fix it, so the kernel is not merely
  reading past `cache_seqlens` into garbage.
- **Not the engine's Context construction.** The second half of this script
  drives `flash_attn_with_kvcache` directly with random tensors and no engine,
  and reproduces the nondeterminism there.

That leaves a race inside the decode kernel itself
(`csrc/mfa/decode.cuh`, `flash_attention_fwd_split_kv_kernel`). The pattern —
clean at low occupancy, corrupting unpredictably as more thread blocks are
resident, severity varying run to run — is the signature of a missing or
mis-scoped barrier on shared memory. Note the kernel aliases several shared
buffers over the same `extern __shared__ char smem_data[]` region
(`decode.cuh:588` reuses the Q/score region for `warp_max_val`, `:628` for
`warp_output`) while the launch in `csrc/mfa/flash.cu` sizes that allocation
for Q + K + V only. Confirming the exact barrier is a task for whoever fixes the
backend; this script's job is to make the failure reproducible.

Consequences for this repo:

- `max_num_batched_seqs` above ~5 is not currently safe. The 4 GB profile in
  `experiments/engine_spec_gate.py` uses 8; the stock `Config` uses 512.
- Existing gates pass because they run two prompts, i.e. batch width 2.
- A speculative verify pass has batch width `num_requests * (K+1)`, so a single
  request at K=4 sits at width 5 — the last safe width — and K=8 does not.
  `experiments/verify_pass_gate.py` is gated at K=4 for this reason.

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
    if first_bad:
        print(f"RESULT: decode is nondeterministic from batch width {first_bad} upward.")
        print(f"        The kernel reproduces it without the engine: {isolated}.")
        print("        Treat max_num_batched_seqs > "
              f"{first_bad - 1} as unsafe until the backend is fixed.")
    else:
        print("RESULT: decode was deterministic at every width tested. If this "
              "follows a backend rebuild, the race may be fixed -- re-check "
              "the widths documented at the top of this file.")
    # This script reports; it is not a pass/fail gate, because the bug it
    # documents is not this project's to fix.
    return 0


if __name__ == "__main__":
    sys.exit(main())
