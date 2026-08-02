# Speculative Decoding Progress

Last updated: 2026-08-03

This file is a concise handoff log for LLM agents. It tracks progress against [speculative-decoding-plan.md](speculative-decoding-plan.md) and should be updated as phases advance.

## Current Status

- **Phase 0: closed as a GO.** The earlier no-go rested on the assumption that
  the verify pass had to use the decode kernel. It does not — a verify pass is
  a chunked-prefill shape and routes through `flash_attn_varlen_func` with a
  `block_table`. No backend patch needed.
- **Phase 2: complete and passing.**
  [experiments/spec_decode_prototype.py](experiments/spec_decode_prototype.py)
  implements draft-and-verify on plain HF models; both gates pass. Results
  below.
- Phase 1 (draft model training) is **deliberately deferred** — see ordering
  note below.
- **Phase 3 onward is blocked on the environment**, not on the design. The
  engine cannot run on this host yet; see Environment State.

## Phase 2 Results (2026-08-03, CPU, Qwen3-0.6B float32)

Run with `python experiments/spec_decode_prototype.py`. Needs torch +
transformers only — no mini-flash-attention, no flashinfer, no triton, no GPU.

**Gate 1 — the sampler returns the target's distribution.** 40k trials against
synthetic mismatched p/q, comparing the empirical distribution of the first
returned token against p: worst deviation 2.07 sigma at K=1, 1.18 sigma at
K=4. This is the Leviathan/Chen lemma checked directly on the real function.

**Gate 2 — greedy exact match.** 3 prompts x K in {2,4,8} x 2 draft models =
**18/18 runs token-for-token identical to plain greedy decoding.**

| draft | K | acceptance | tokens/round |
|---|---|---|---|
| random-init tiny Qwen3 | 2 / 4 / 8 | 0.0% everywhere | 1.00 |
| target drafting for itself @ T=0.8 | 2 | 44–77% | 1.88–2.46 |
| target drafting for itself @ T=0.8 | 4 | 25–81% | 1.88–4.00 |
| target drafting for itself @ T=0.8 | 8 | 23–58% | 2.67–5.33 |

The two draft models were chosen to cover the failure modes, not to be
realistic: the random draft is rejected at i=0 in every single round (the edge
case implementations usually get wrong), while the self-draft produces a mix of
partial acceptance, full acceptance, and bonus-token rounds. Acceptance rate
falls as K grows, exactly the tradeoff Phase 7's sweep is meant to chart.

Design points worth carrying into the engine:

- Temperature 0 is represented as a **one-hot distribution**, not as a branch
  in the sampler. Greedy therefore runs through the identical rejection-sampling
  code path as temperature > 0, which is what makes the exact-match test a real
  test of the general path rather than of a shortcut.
- `rejection_sample(draft_tokens, q, p, generator) -> (tokens, num_accepted)` is
  a pure function with no model or cache dependencies. Phase 5 should port it
  into `Sampler` as-is.
- A round **always returns `num_accepted + 1` tokens** — the residual resample
  on rejection, or the bonus token on full acceptance. That invariant is what
  guarantees forward progress when the draft is useless (visible above as the
  random draft's 1.00 tokens/round at 0% acceptance).
- Cache invariant: both caches hold KV for `tokens[:cached]` with
  `cached <= len(tokens) - 1`, so the last token is always unprocessed.
  `DynamicCache.crop()` discards rejected KV. Phase 4 must reproduce this
  inside the paged cache, where `crop` does not exist.

## Environment State (verified 2026-08-03)

Partially built. Standalone experiments run; the engine does not.

- `.venv/` has `torch==2.9.1+cu130`, `transformers==4.57.3`, tokenizers,
  safetensors, numpy, xxhash, huggingface_hub. Missing for the engine:
  `mini-flash-attention`, `triton`, `flashinfer`.
- **`~/huggingface/Qwen3-0.6B/` is downloaded** (1.5 GB safetensors + tokenizer).
- **CUDA is not available.** Host driver is **566.07** (CUDA 12.7 era); the
  installed torch is a **cu130** build, which needs a newer driver.
  `torch.cuda.is_available()` returns `False` with `cudaErrorNotSupported`.
  The fix is to reinstall torch from the **cu126** index (or update the
  NVIDIA driver past 580). Phase 2 was run on CPU, which is fine — its gates
  are device-independent.
- GPU hardware: RTX 3050 Laptop, 4096 MiB total, ~3600 MiB free.

Known Windows build risks, still none attempted: the from-source CUDA build of
`mini-flash-attention`, the lack of official Windows `triton` wheels, and the
lack of a `flashinfer` Windows wheel (see the note in CLAUDE.md about
`flashinfer` being missing from `requirements.txt` entirely).

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
- **The rejection-sampling maths is correct and implemented** — see Phase 2
  Results. Any later divergence from greedy is an engine-integration bug, not
  an algorithm bug. That separation is the whole point of doing Phase 2 first.

## Deviation from the plan, and why

**Phase 1 (train the draft model) is deferred until after Phase 6.**

Rejection sampling returns the target model's exact distribution regardless of
draft quality — a poor draft lowers the acceptance rate but can never change
the output tokens. So the entire correctness effort (Phases 2, 4, 5, 6) can be
driven by a randomly-initialised tiny Qwen3 sharing the target's vocabulary,
and a trained draft is only needed for Phase 7's speedup numbers.

Phase 2 confirms this empirically: the random-init draft achieved 0% acceptance
and still reproduced greedy output exactly, across all three K values.

## Blockers

- **Phase 3+ needs a runnable engine**, which needs (in order): a cu126 torch
  reinstall to get CUDA working at all, then `triton-windows`, then a
  from-source `mini-flash-attention` build, then either `flashinfer` or a lazy
  import in `sampler.py`. None of these are attempted yet, and each is an
  independent Windows-specific risk.
- No blockers on the design or the algorithm.

## Recommended Next Step

Two independent tracks; the second does not depend on the first.

1. **Unblock the engine** (this is the long pole):
   - Reinstall torch from the cu126 index and confirm `torch.cuda.is_available()`.
   - Make `flashinfer` a lazy import in `minivllm/models/layers/sampler.py` —
     it is only needed for top-k/top-p, and the correctness harness is greedy.
   - Install `triton-windows`, then build `mini-flash-attention` from source.
   - Sanity-check the engine end to end with `python run.py` before touching
     any speculative code.
2. **Phase 3 (config + draft loading)** can be written now but not executed:
   add `use_speculative_decoding`, `draft_model`, `num_speculative_tokens` to
   `Config`, and load the draft in `Executor.__init__`. Low risk, but it
   cannot be validated until track 1 lands, so it is worth keeping small.

Then Phase 4, whose first job is the two empirical paged-varlen checks listed
at the end of `docs/spec_decoding_feasibility.md`, plus the `block_table=` bug
fix in `attention.py`.

## Phase Checklist

- Phase 0: **complete — go decision recorded**, verify path identified.
  Committed as `8f16197` on branch `test`.
- Phase 1: deferred by design (see above); random-init draft used meanwhile.
- Phase 2: **complete — both gates passing**, 18/18 exact match. See results above.
- Phase 3: not started; can be written before the environment is fixed.
- Phase 4: unblocked in design — route verify through `flash_attn_varlen_func`
  + `block_table`; blocked in practice on the environment.
- Phase 5: not started. Port `rejection_sample` from the Phase 2 prototype.
- Phase 6: not started.
- Phase 7: needs a trained draft model (Phase 1) to be meaningful.

## Handoff Notes for Agents

- Read [speculative-decoding-plan.md](speculative-decoding-plan.md) first.
- [docs/spec_decoding_feasibility.md](docs/spec_decoding_feasibility.md) is the
  authoritative note; it records the go decision and the two remaining
  empirical checks on paged-varlen semantics.
- The greedy exact-match harness is the correctness gate at every level —
  prototype (Phase 2, done) and engine (Phase 6, pending).
- Do not modify paging or scheduler core code unless the plan is explicitly revised.
