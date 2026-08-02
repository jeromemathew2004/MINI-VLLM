# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

mini-vllm is a lightweight LLM inference engine built from scratch for learning and experimentation (inspired by vLLM and nano-vllm). The implementation is intentionally small so the main execution flow (scheduling → KV-cache allocation → batched forward pass → sampling) stays easy to trace end-to-end. Qwen3 is the primary supported model; Gemma3 is also present in the model tree.

## Requirements & environment

- Python 3.10+, CUDA-capable GPU required (the executor hardcodes `torch.set_default_device("cuda")`).
- Dependencies: `pip install -r requirements.txt`. Note `mini-flash-attention` is installed from GitHub (`w4096/mini-flash-attention`), not PyPI — it is imported as `mini_flash_attention` in `minivllm/models/layers/attention.py`.
- Demo/benchmark scripts default to a local model at `~/huggingface/Qwen3-0.6B/`; download with `hf download Qwen/Qwen3-0.6B`.
- A `.venv` already exists in the repo root.

## Common commands

```sh
# Interactive chat REPL
python chat.py --model ~/huggingface/Qwen3-0.6B/

# Scripted batch-generation demo
python run.py

# Throughput benchmarks
python benchmark/run_mini_vllm.py
python benchmark/run_vllm.py   # compares against real vLLM

# Tests (pytest, no config file — just point at the tests dir)
pytest tests/
pytest tests/test_block_manager.py -v
```

There is no lint/format tooling configured in this repo.

## Architecture

Request flow: `Engine.step()` drives one iteration — `Scheduler.schedule()` picks a `Batch`, `Executor.execute()` runs it on the GPU and samples tokens, `Scheduler.update()` appends the tokens to requests and frees/advances KV-cache blocks.

- **`minivllm/config/config.py`** — `Config` dataclass holds model path, HF config, scheduler limits (`max_num_batched_tokens`, `max_num_batched_seqs`), and KV-cache sizing (`kv_cache_num_blocks`, `kv_cache_block_size`, `gpu_memory_utilization`). `kv_cache_num_blocks` gets overwritten at runtime by `Executor._init_kv_cache` based on actual free GPU memory.
- **`minivllm/engine/engine.py`** — top-level `Engine` class; owns tokenizer, `Executor`, `Scheduler`, `Metrics`. `generate()` is the blocking batch-generation API used by `run.py`/benchmarks; `submit()` + `step()` is the incremental API used by `chat.py` for streaming.
- **`minivllm/engine/request.py`** — `Request` tracks prompt + generated tokens, KV-cache block ids (`req.blocks`), and lifecycle state (`WAITING` → `RUNNING` → `FINISHED`).
- **`minivllm/scheduler/scheduler.py`** — continuous-batching scheduler. Always prefers scheduling a prefill batch over a decode batch (`schedule()` tries `_schedule_prefill` first). Handles preemption: if KV-cache blocks run out during decode scheduling, it evicts the most-recently-added running request back to `waiting` and frees its blocks (`preempt`).
- **`minivllm/kvcache/block_manager.py`** — `KVCacheBlockManager` implements paged KV-cache allocation (fixed-size blocks, free-list + refcounts) and optional prefix caching keyed by rolling `xxhash` of block token content (`support_prefix_cache`, off by default). `Request.blocks` holds a list of block ids; the last block is often partially filled.
- **`minivllm/executor/executor.py`** — `Executor` owns the model, allocates the physical KV-cache tensor (`torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)`) and wires per-layer `k_cache`/`v_cache` slices into attention modules. Builds prefill vs decode inputs differently:
  - Prefill uses varlen flash-attention (`cu_seqlens_q/k`, optional `block_table` when there's cached prefix).
  - Decode builds one-token-per-request inputs (`_build_decode_input`) and uses `flash_attn_with_kvcache`, which asserts `seqlen_q == 1`. That assert constrains *this* path only — multi-token passes go through `flash_attn_varlen_func` instead, which takes a paged `block_table` and has no such limit. See `docs/spec_decoding_feasibility.md`.
  - Runs a `_warmup_model()` prefill pass at startup to size the KV cache correctly before real allocation.
- **`minivllm/executor/context.py`** — `Context` dataclass carries all per-forward-pass tensors (positions, slot_mapping, cu_seqlens, block_table, cache_seqlens) threaded through every model/attention call; distinguishes prefill vs decode via `ctx.prefill`.
- **`minivllm/executor/graph.py`** — `CudaGraphRunner` pre-captures CUDA graphs for a fixed set of decode batch sizes (`[1, 2, 4, 8, 16, 32, ...]` up to `max_batch_size` capped at 256) to cut decode-step launch overhead; only used for decode, never prefill.
- **`minivllm/models/`** — `loader.py` maps HF `architectures[0]` to a model class via `models/models/__init__.py`'s `_MODELS` registry, then loads safetensors weights with a custom `weight_loader` mechanism for packed QKV projections (see `Qwen3ForCausalLM.load_weights`). Model `forward()` signatures are uniformly `(ctx: Context, input_ids, positions) -> logits`, and at prefill time only the last hidden state per sequence is projected to logits (`ctx.cu_seqlens_q[1:] - 1`).
- **`minivllm/models/layers/attention.py`** — `FlashAttention` wraps `mini_flash_attention`'s `flash_attn_varlen_func` (prefill) and `flash_attn_with_kvcache` (decode), and writes new K/V into the paged cache via a Triton kernel (`store_kvcache_kernel`) before attention runs.
- **`minivllm/engine/metrics.py`** — tracks prefill/decode throughput, TTFT, inter-token latency; surfaced live in `chat.py`'s progress bar and `Engine.generate`'s tqdm postfix.

### Speculative decoding effort (in progress, not yet implemented)

`speculative-decoding-plan.md` and `PROGRESS.md` track the speculative decoding effort. Current status: **Phase 0 complete, go decision recorded (2026-08-03).** An earlier no-go was reversed — it had assumed the multi-token verify pass must use the decode kernel, but a verify pass is a chunked-prefill shape and routes through `flash_attn_varlen_func` with a `block_table`, so no backend patch is needed. `docs/spec_decoding_feasibility.md` is the authoritative note.

Two things to know before working on this:

- **Known latent bug:** `attention.py` never passes `block_table=` to `flash_attn_varlen_func`, despite the adjacent comment saying it must. Harmless today (`support_prefix_cache=False` by default) but it is a prerequisite fix for the verify pass.
- **Phase 1 (draft model training) is deliberately deferred** until correctness is locked. Rejection sampling is exactly distribution-preserving regardless of draft quality, so a random-init tiny draft drives the correctness harness; a trained draft only affects acceptance rate and speedup.

Do not modify scheduler/paging core code — speculative decoding is a decode-path feature. Check `PROGRESS.md` for the latest phase status first.
