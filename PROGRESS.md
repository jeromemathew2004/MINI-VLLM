# Speculative Decoding Progress

Last updated: 2026-08-03

This file is a concise handoff log for LLM agents. It tracks progress against [speculative-decoding-plan.md](speculative-decoding-plan.md) and should be updated as phases advance.

## Current Status

- **Phase 0 re-opened and closed as a GO.** The earlier no-go rested on the
  assumption that the verify pass had to use the decode kernel. It does not —
  a verify pass is a chunked-prefill shape and routes through
  `flash_attn_varlen_func` with a `block_table`. No backend patch needed.
- Phase 1 (draft model training) is **deliberately deferred** — see ordering
  note below.
- Environment build is in progress; the workspace previously had an empty venv
  and no model artifacts.

## What Has Been Confirmed

- `Executor._build_decode_input` decodes one token per request per step.
- `FlashAttention.forward` stores KV entries during decode and calls
  `flash_attn_with_kvcache(q.unsqueeze(1), ...)`.
- `Sampler` only exposes the existing greedy / top-k / top-p path.
- `Config` does not yet contain speculative-decoding fields.
- `flash_attn_with_kvcache` asserts `seqlen_q == 1` — confirmed, unchanged.
- **`flash_attn_varlen_func` has no `seqlen_q` restriction, accepts a paged
  `block_table`, and supports GQA.** This is the verify path.
- **Latent bug:** `attention.py` never passes `block_table=` to
  `flash_attn_varlen_func` despite the comment saying it must. Dead code today
  (`support_prefix_cache=False` by default), but must be fixed for Phase 4.
- Host GPU: RTX 3050 Laptop, 4096 MiB total, ~3055 MiB free.

## Deviation from the plan, and why

**Phase 1 (train the draft model) is deferred until after Phase 6.**

Rejection sampling returns the target model's exact distribution regardless of
draft quality — a poor draft lowers the acceptance rate but can never change
the output tokens. So the entire correctness effort (Phases 2, 4, 5, 6) can be
driven by a randomly-initialised tiny Qwen3 sharing the target's vocabulary,
and a trained draft is only needed for Phase 7's speedup numbers.

Doing correctness first means the training run is validated by a harness that
already works, instead of debugging both at once.

## Blockers

- None on the design. Remaining risk is environment build, not algorithm.

## Environment Requirements

- Target model: Qwen3-0.6B at `~/huggingface/Qwen3-0.6B/`
  (`Config.__post_init__` asserts the path is a directory).
- `requirements.txt` is missing `flashinfer`, which `sampler.py` imports at
  module top level. flashinfer has no reliable Windows wheel; since the
  correctness harness runs greedy, making that import lazy is the mitigation.
- `triton` is needed only for `store_kvcache_kernel`; official triton has no
  Windows wheels (`triton-windows` is the usual substitute).
- `mini-flash-attention` is a from-source CUDA build and needs a CUDA toolkit
  plus MSVC.

## Recommended Next Step

1. Finish environment build; download Qwen3-0.6B.
2. Phase 2: standalone draft+verify prototype under `experiments/`, using
   plain HF calls only. Needs torch + transformers only — no
   mini-flash-attention, no flashinfer, no triton.
3. Gate on the greedy exact-match test before any engine integration.

## Phase Checklist

- Phase 0: **complete — go decision recorded**, verify path identified.
- Phase 1: deferred by design (see above); random-init draft used meanwhile.
- Phase 2: next up.
- Phase 3: not started.
- Phase 4: unblocked — route verify through `flash_attn_varlen_func` + `block_table`.
- Phase 5: not started.
- Phase 6: not started.
- Phase 7: needs a trained draft model (Phase 1) to be meaningful.

## Handoff Notes for Agents

- Read [speculative-decoding-plan.md](speculative-decoding-plan.md) first.
- [docs/spec_decoding_feasibility.md](docs/spec_decoding_feasibility.md) is the
  authoritative note; it now records the go decision and the two remaining
  empirical checks on paged-varlen semantics.
- Keep the greedy exact-match harness as the correctness gate.
- Do not modify paging or scheduler core code unless the plan is explicitly revised.
