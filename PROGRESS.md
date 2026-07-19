# Speculative Decoding Progress

Last updated: 2026-07-19

This file is a concise handoff log for LLM agents. It tracks progress against [speculative-decoding-plan.md](speculative-decoding-plan.md) and should be updated as phases advance.

## Current Status

- Phase 0 is complete enough to make a decision.
- The current backend blocks the planned multi-token verify path.
- The repo is still at the no-code feasibility stage for speculative decoding integration.

## What Has Been Confirmed

- `Executor._build_decode_input` currently decodes one token per request per step.
- `FlashAttention.forward` stores KV entries during decode and uses `flash_attn_with_kvcache(q.unsqueeze(1), ...)`.
- `Sampler` only exposes the existing greedy / top-k / top-p path.
- `Config` does not yet contain speculative-decoding fields.
- The upstream `mini-flash-attention` implementation asserts `seqlen_q == 1` for KV-cache decode.
- The host GPU is an NVIDIA GeForce RTX 3050 Laptop GPU with 4096 MiB of VRAM.
- A live `nvidia-smi` reading reported 2641 MiB free and 1324 MiB used.

## Completed Work

- Updated [README.md](README.md) with install, run, benchmark, and chat command details.
- Wrote [docs/spec_decoding_feasibility.md](docs/spec_decoding_feasibility.md) to record Phase 0 findings and the backend blocker.
- Verified the live repo code paths that control decode, sampling, and KV-cache writes.
- Inspected the upstream `mini-flash-attention` source and confirmed the single-token KV-cache decode limitation.

## Blockers

- `mini-flash-attention` does not currently support multi-token KV-cache verify passes.
- The workspace does not currently contain local Qwen3 model artifacts, so the plan's exact simultaneous model-load check was not exercised here.

## Recommended Next Step

1. Decide whether to patch `mini-flash-attention` to support `seqlen_q > 1` or to redesign speculative decoding around repeated single-token verify calls.
2. If the backend is patched, proceed to Phase 2 with the standalone draft+verify prototype.
3. If the backend is not patched, treat the current feasibility note as the stop point and do not start engine integration.

## Phase Checklist

- Phase 0: feasibility note written, backend blocker identified.
- Phase 1: not started.
- Phase 2: not started.
- Phase 3: not started.
- Phase 4: blocked by backend limitation.
- Phase 5: not started.
- Phase 6: not started.
- Phase 7: not started.

## Handoff Notes for Agents

- Read [speculative-decoding-plan.md](speculative-decoding-plan.md) first.
- Use [docs/spec_decoding_feasibility.md](docs/spec_decoding_feasibility.md) as the authoritative blocker summary.
- Keep the greedy exact-match harness as the correctness gate once prototype work starts.
- Do not modify paging or scheduler core code unless the plan is explicitly revised.
