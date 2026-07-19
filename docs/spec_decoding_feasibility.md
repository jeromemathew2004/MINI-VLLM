# Speculative Decoding Feasibility Note

Date: 2026-07-19

## Confirmed from repo code

- `Executor._build_decode_input` in [minivllm/executor/executor.py](../minivllm/executor/executor.py) builds one decode token per request per step.
- `FlashAttention.forward` in [minivllm/models/layers/attention.py](../minivllm/models/layers/attention.py) stores KV entries on every decode call and currently invokes `flash_attn_with_kvcache(q.unsqueeze(1), ...)`, which is a single-query decode shape.
- `Sampler` in [minivllm/models/layers/sampler.py](../minivllm/models/layers/sampler.py) only exposes the plain greedy / top-k / top-p sampling path today.
- `Config` in [minivllm/config/config.py](../minivllm/config/config.py) does not yet contain speculative-decoding fields.

## Unverified external assumption

- The installed `mini_flash_attention` backend is not present in the workspace virtual environment.
- The upstream `mini-flash-attention` source does not implement multi-token KV-cache decode: `flash_attn_with_kvcache` asserts `seqlen_q == 1` in `mini_flash_attention/interface.py`, and the decode kernel is specialized for single-token queries in `csrc/mfa/flash.cu`.

## VRAM check

- The host GPU reported by Windows is an NVIDIA GeForce RTX 3050 Laptop GPU with 4096 MiB of VRAM.
- Live `nvidia-smi` output at the time of this check reported 2641 MiB free and 1324 MiB used.
- The workspace does not currently have local Qwen3 model artifacts, so the exact simultaneous load test from the plan was not executed here; the measured free memory still indicates limited but non-zero headroom.

## Decision

- Go decision for engine integration: **no-go for the planned Phase 4 multi-token verify path** with the current backend.
- The repo code already shows the speculative-decoding integration points, but the backend assumption is the blocking item. A viable path now requires either patching `mini-flash-attention` to accept `seqlen_q > 1` or redesigning the verify step around repeated single-token decode calls.

## Next validation step

1. Decide whether to patch `mini-flash-attention` itself or to replace the multi-token verify step with a repeated single-token fallback.
2. Once that decision is made, proceed to the standalone prototype phase and preserve the greedy exact-match harness.
