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

## Phase 3 Status — written, NOT yet executed

All engine-side Phase 3 code is written and syntax-checked only. **No part of
it has run**, because the engine cannot start on this host yet. Treat every
claim below as unverified until `python run.py` works.

- `Config` gains `use_speculative_decoding`, `draft_model`,
  `num_speculative_tokens`, and a derived `draft_hf_config`.
  `__post_init__` asserts the draft's vocab size equals the target's (a
  mismatch is silent corruption, not a crash), forces the draft's dtype to the
  target's, and now runs `os.path.expanduser` on both model paths — the
  defaults are written with a leading `~`, which only a unix shell expands.
- `loader.py` gains `load_draft_model()`. `_initialize_model` now takes an
  `hf_config` rather than a `Config`, so one code path serves both models.
- `Executor.__init__` loads the draft **before** `_warmup_model()`, so the
  free/peak memory figures `_init_kv_cache` reads already include the draft's
  weights. Note the warmup pass still exercises the target only, so the
  draft's *activation* peak is unaccounted for — revisit in Phase 4, when the
  draft actually runs a forward pass.
- `_init_kv_cache` was factored into `_kv_block_bytes` / `_alloc_kv_cache` /
  `_wire_kv_cache`, and charges both models' per-block cost against the same
  memory budget.
- `Sampler`'s `flashinfer` import is now lazy (inside the top-k/top-p branch).

### Draft cache: deviation from the plan, and why

The plan called for a non-paged contiguous draft cache with no block manager
involvement. **Implemented instead: the draft gets its own cache tensor,
allocated with the same block size and the same block count as the target's.**

`FlashAttention.forward` has no non-paged path — it always writes through
`ctx.slot_mapping` into a block-indexed cache and passes `ctx.block_table` to
the decode kernel. A contiguous draft cache would have meant adding a second
attention path to a file the target model also depends on.

Sharing the block ids avoids that entirely. A slot index is
`block_id * kv_cache_block_size + offset`, which depends only on the block
size, so the ids in `req.blocks` — handed out by the single
`KVCacheBlockManager` — address both caches. `ctx.slot_mapping` and
`ctx.block_table` are valid for the draft unchanged, and the block manager is
untouched, which respects the "don't modify paging core" constraint better
than a parallel allocator would. Cost is roughly 1/56 of the target's cache
for a 4-layer / 2-KV-head / 64-dim draft against Qwen3-0.6B's 28 / 8 / 128.

### Draft checkpoint

`experiments/make_random_draft.py` writes a random-init tiny Qwen3 to
`~/huggingface/Qwen3-draft-random/`. **This script has been run**: 42.0M
parameters total, 3.1M non-embedding, tied embeddings, and it self-checks that
`architectures` is `Qwen3ForCausalLM`, that the vocab size matches, and that
its tokenizer produces identical ids to the target's.

Smaller than the 10-30M non-embedding the plan specifies for the *trained*
draft — this one exists to make the correctness harness fast, not to be
accepted often. Dimensions are CLI flags.

### Phase 3 exit criterion, still outstanding

Engine loads both models and non-speculative output is byte-identical with
`use_speculative_decoding` on and off. Cannot be run until the environment is
built.

## Environment State (verified 2026-08-03)

Partially built. Standalone experiments run; the engine does not.

- `.venv/` has `transformers==4.57.3`, tokenizers, safetensors, numpy, xxhash,
  huggingface_hub. Missing for the engine: `mini-flash-attention`, `triton`,
  `flashinfer`.
- **torch: mid-migration.** The venv had `2.9.1+cu130`, which the host driver
  cannot run. A `pip install --index-url .../cu126 --force-reinstall --no-deps
  torch==2.9.1+cu126` was launched and **had not finished when the session
  ended — its result is unverified.** Check with
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
  and re-run the install if it still reports `cu130` or `False`.
- **`~/huggingface/Qwen3-0.6B/` is downloaded** (1.5 GB safetensors + tokenizer).
- **`~/huggingface/Qwen3-draft-random/` is written** — see Draft checkpoint above.
- Host driver is **566.07** (CUDA 12.7 era). It natively supports CUDA 12.6, so
  cu126 needs no driver update. cu130 would require a driver past 580.
- GPU hardware: RTX 3050 Laptop, compute capability **8.6**, 4096 MiB total,
  ~3600 MiB free.
- Phase 2 was run on CPU, which is fine — its gates are device-independent.

### GPU suitability — settled 2026-08-03

`mini-flash-attention` requires **SM 80+** ("NVIDIA Ampere GPU or newer",
CUDA 11.8+, tested on CUDA 12.8). The host GPU reports **compute capability
8.6**, so it qualifies.

This also rules out the obvious escape hatch: **free Colab and free Kaggle
both provide a T4 (sm_75)**, and Kaggle's other free option is a P100
(sm_60). Both are *below* the SM 80 floor that this laptop clears. Any cloud
fallback would have to be a paid A100/L4/L40S tier. The local GPU is the only
free environment that meets the requirement.

### Remaining Windows build risk

The GPU is not the risk; the toolchain is.

- `mini-flash-attention`'s `setup.py` compiles CUTLASS with `-std=c++20`, and
  its README documents a Linux-only build
  (`build/lib.linux-x86_64-cpython-312`). The MSVC path is unexercised. It
  passes no explicit `-gencode` flags, relying on torch's `CUDAExtension` to
  target the detected arch, and it needs `$CUDA_HOME/include/cccl`.
- Building it needs a **CUDA Toolkit** and **MSVC**, neither of which is
  installed on this host (no `nvcc`, no Visual Studio).
- Official `triton` ships no Windows wheels; `triton-windows` is the
  substitute. Needed for `store_kvcache_kernel` *and* for the `@torch.compile`
  on `MLP.forward`.
- `flashinfer` has no reliable Windows wheel. Mitigated: its import is now
  lazy, inside the top-k/top-p branch of `Sampler.forward`, so greedy decoding
  — which every correctness gate uses — no longer depends on it.
- 4 GB VRAM stays tight. `max_model_len` and `gpu_memory_utilization` will
  need lowering for local runs.

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

- **Phase 3+ needs a runnable engine.** Nothing in `minivllm/` can be imported
  on this host: `attention.py` imports `triton` and `mini_flash_attention` at
  module scope, and `Executor` hardcodes `torch.set_default_device("cuda")`.
- No blockers on the design or the algorithm.

# RESUME HERE

This is the runbook for the next session. Steps 0-3 are environment work;
step 4 is where Phase 3 actually gets validated. **Do not skip step 3** —
proving the engine works *without* speculative decoding first is what keeps a
backend failure from being misread as a rejection-sampling bug.

### Step 0 — prerequisites the user installs by hand

These are GUI/admin installers; an agent cannot do them. Install in this
order, so the CUDA installer can register its MSBuild integration:

1. **Visual Studio Build Tools 2022**, workload "Desktop development with
   C++" (MSVC v143 + Windows SDK). Provides `cl.exe`.
2. **CUDA Toolkit 12.6**, matching the cu126 torch build. Driver 566.07
   supports 12.6 natively — no driver update needed.

Verify: `nvcc --version` reports 12.6, and `cl.exe` resolves from a
Developer Command Prompt (or after running `vcvars64.bat`).

### Step 1 — finish the torch migration

```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Must print a `+cu126` version and `True`. If it still says `cu130` or `False`,
the background install from the previous session did not complete:

```sh
pip install --index-url https://download.pytorch.org/whl/cu126 \
    --force-reinstall --no-deps "torch==2.9.1+cu126"
```

Note plain `torch==2.9.1` is **not** enough — pip treats the installed
`2.9.1+cu130` as already satisfying it and silently does nothing. The
`+cu126` local version and `--force-reinstall` are both required.

### Step 2 — the remaining engine dependencies

```sh
pip install triton-windows        # official triton has no Windows wheels
pip install git+https://github.com/w4096/mini-flash-attention.git
```

`triton` is needed for `store_kvcache_kernel` *and* for the `@torch.compile`
on `MLP.forward`. `mini-flash-attention` is the risky one — see Remaining
Windows build risk above. `flashinfer` is **not** required: its import is now
lazy and only greedy decoding is exercised by the correctness gates.

Verify: `python -c "import triton, mini_flash_attention"`.

### Step 3 — prove the engine runs at all, before any spec-decode work

```sh
python run.py
```

Expect to tune for 4 GB VRAM: lower `max_model_len` and possibly raise
`gpu_memory_utilization` in `Config`. A clean generation here is the
precondition for everything below.

### Step 4 — validate Phase 3 (the code is written; nothing has run)

Its exit criterion is that loading the draft changes nothing yet:

1. `Config(use_speculative_decoding=True, draft_model="~/huggingface/Qwen3-draft-random/")`
   constructs without tripping the vocab or path asserts.
2. The engine starts, logs both "Loading model on device..." and "Loading
   draft model on device...", and logs the draft cache allocation.
3. **Greedy output is byte-identical with the flag on and off.** Phase 3 adds
   no behaviour — the draft is loaded and then ignored.

Watch for: `kv_cache_num_blocks` shrinking once the draft's per-block cost is
charged to the same budget, and `assert kv_cache_num_blocks > 0` failing if
4 GB proves too tight.

Also worth running once CUDA works, independent of the engine —
`Config.__post_init__` imports no torch, so this can be checked even before
step 2 lands:

```sh
python -c "from minivllm.config.config import Config; \
c = Config(use_speculative_decoding=True, draft_model='~/huggingface/Qwen3-draft-random/'); \
print(c.draft_hf_config.vocab_size, c.draft_hf_config.dtype, c.num_speculative_tokens)"
```

### Step 5 — Phase 4

First tasks, in order:

1. Fix the `block_table=` bug in `attention.py` (it is never passed to
   `flash_attn_varlen_func` despite the comment saying it must be). This is a
   real pre-existing correctness fix and a prerequisite for the verify pass.
2. Run the two empirical paged-varlen checks listed at the end of
   `docs/spec_decoding_feasibility.md`.
3. Then build the multi-token verify pass itself.

## Phase Checklist

- Phase 0: **complete — go decision recorded**, verify path identified.
  Committed as `8f16197` on branch `test`.
- Phase 1: deferred by design (see above); random-init draft used meanwhile.
- Phase 2: **complete — both gates passing**, 18/18 exact match. See results above.
- Phase 3: **code written, never executed.** Config fields, `load_draft_model`,
  dual KV-cache allocation, draft checkpoint generated. Validate at step 4 of
  the runbook before calling it done.
- Phase 4: unblocked in design — route verify through `flash_attn_varlen_func`
  + `block_table`; blocked in practice on the environment.
- Phase 5: not started. Port `rejection_sample` from the Phase 2 prototype.
- Phase 6: not started.
- Phase 7: needs a trained draft model (Phase 1) to be meaningful.

## Handoff Notes for Agents

- **Start at "RESUME HERE" above.** Steps 0-3 are environment, step 4 validates
  the Phase 3 code that is already written but has never run.
- Read [speculative-decoding-plan.md](speculative-decoding-plan.md) first.
- [docs/spec_decoding_feasibility.md](docs/spec_decoding_feasibility.md) is the
  authoritative note; it records the go decision and the two remaining
  empirical checks on paged-varlen semantics.
- The greedy exact-match harness is the correctness gate at every level —
  prototype (Phase 2, done) and engine (Phase 6, pending).
- Do not modify paging or scheduler core code unless the plan is explicitly revised.
