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
- **Phase 3: complete and validated on GPU (2026-08-03).** The environment is
  built, the engine runs, and the exit criterion passes byte-identically. See
  Phase 3 Validation.
- **Phase 4: complete and gated on GPU (2026-08-03).** The multi-token verify
  pass exists (`Executor.verify`) and provably equals K+1 sequential decode
  steps. It is **not** built on `flash_attn_varlen_func` as the plan and the
  Phase 0 note assumed — that kernel's causal mask is top-left anchored, which
  makes the shape unusable. See Phase 4 Results.
- **Two bugs found along the way**, one fixed here and one out of scope:
  prefix-caching prefill was doubly broken (missing `block_table` *and* the
  wrong mask anchoring), and **plain decode is nondeterministic at batch width
  >= 6** — pre-existing, unrelated to speculative decoding, and now the main
  blocker on this project's usable K and on serving generally.
- **Phase 5 is next.** Start at RESUME HERE.

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

## Phase 3 Validation (2026-08-03, RTX 3050 4 GB, CUDA 12.6)

**Executed and passing.** The code below is no longer "written but unrun".

Gate: [experiments/engine_spec_gate.py](experiments/engine_spec_gate.py)

```
python experiments/engine_spec_gate.py --compare --cuda-graph
```

builds the engine twice in one process — once plain, once with
`use_speculative_decoding=True` — and diffs the greedy token ids.

- Both prompts, 64 tokens each: **token-for-token identical**, matching
  SHA-256. Phase 3 adds no behaviour, and measurably does not.
- Both models load ("Loading model on device...", "Loading draft model on
  device...") and the draft cache is allocated over the same block ids.
- `kv_cache_num_blocks` **237 → 184** once the draft's per-block cost is
  charged against the same budget. This is the shrink the old runbook flagged
  to watch for; it is correct behaviour, not a regression.
- Verified identical across all three of: eager, CUDA graphs on, and draft
  loaded. CUDA-graph decode output equals eager output exactly.

Engine throughput for reference (0.6B, bf16, 4 GB laptop GPU): ~20 tok/s
decode, TTFT ~7 s on a cold prefill. Slow, but this phase is about
correctness — Phase 7 is where speed matters.

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

### Phase 3 exit criterion — MET

Engine loads both models and non-speculative output is byte-identical with
`use_speculative_decoding` on and off. Verified 2026-08-03; see Phase 3
Validation above.

## Phase 4 Results (2026-08-03, RTX 3050 4 GB)

Two gate scripts, both passing:

```
python experiments/paged_varlen_check.py     # kernel semantics -> RESULT: PASS
python experiments/verify_pass_gate.py       # the verify pass  -> GATE: PASS
```

### The plan's verify mechanism was wrong; the phase still lands

`speculative-decoding-plan.md` section 7 and the reversed Phase 0 note both
said the verify pass is a `flash_attn_varlen_func` call with a `block_table`
and `causal=True`. On the GPU that does not work: **the causal mask is
top-left anchored** (`csrc/mfa/prefill.cuh:416` masks `col > row` with no
`seqlen_k - seqlen_q` shift). A K+1-token query against a longer cache would
attend to keys 0..j instead of to its own history. Phase 0 reasoned from the
signature, which does not show this.

**What works instead:** `flash_attn_with_kvcache`. Its `assert seqlen_q == 1`
constrains the *query* dimension, not the token count, so the K+1 tokens ride
in the **batch** dimension as K+1 rows, each with its own `cache_seqlens` and
its own copy of the request's block-table row. Each row then attends to exactly
its own prefix. Still one forward pass, one GEMM per projection; only the
attention gather is per-row. Verified against dense attention at `0.00e+00`.

Full derivation, with the kernel source cited and the block-size rule, is in
[docs/spec_decoding_feasibility.md](docs/spec_decoding_feasibility.md).

### What Phase 4 added

- `Executor._build_verify_input` / `Executor.verify(requests, proposals)`,
  returning `(num_requests, K+1, vocab_size)` logits. Row 0 is what a plain
  decode step would produce; row K is the bonus distribution.
- `block_manager.allocate_block_for_decode(req, extra_tokens=K)` —
  `request_required_blocks` / `can_allocate_new_block` grew the same optional
  argument, defaulting to 0 so existing behaviour is untouched. The allocator
  is now a loop rather than a single `if`, since K can exceed a block.
- `attention.py` now passes `block_table=` to `flash_attn_varlen_func` (the
  documented-but-absent argument) **and** asserts `seqlen_q == seqlen_k` there,
  because passing it is necessary but not sufficient — see above.
- `Config.__post_init__` asserts `kv_cache_block_size % 64 == 0`. Both kernels
  resolve one block-table entry per 64-key tile, so a smaller page silently
  corrupts attention past a sequence's first page. Measured: 16 and 32 corrupt,
  64/128/256 exact.
- CUDA graphs are bypassed for verify passes (plan section 7.4).

### Stale-KV strategy, and how it is tested

The pass writes KV for all K proposals before anyone knows how many are
accepted. If M < K are, the slots for positions `[len+M, len+K)` hold tokens
that never entered the sequence. **They are left in place, not zeroed**: every
read is bounded by `cache_seqlens`, which derives from the committed token
count, and the next round's verify pass starts writing at exactly the first
dead slot. So the region is unreadable until overwritten. This is the
"`cache_seqlens` is the source of truth" option from plan section 7.3.

`verify_pass_gate.py` forces every acceptance regime — full, partial, zero, and
a `mixed` script that cycles M over 0..K so consecutive rounds overwrite stale
regions of every size. Zero-acceptance, the case implementations get wrong
first, is covered every round of its script.

### Why the gate is not "byte-identical greedy output"

It cannot be, and this matters for Phase 6. Model logits depend on the batch
shape: decoding one request at width 1 and at width 5 — no verify pass anywhere
— differs by up to 0.5 absolute (mean 0.087), because cuBLAS picks different
tilings per shape. A verify pass has K+1 rows where decode has 1, so its logits
are *necessarily* not bitwise equal to sequential decode's.

The gate is therefore stated against a measured noise floor:

| gate | claim | result |
|---|---|---|
| 1 | `verify()` with K=0 is one query row, so it must match decode **bitwise** | `max diff = 0.000000` |
| 2 | `verify()` row 0 adds no error beyond its batch width | `max\|decode@5 - verify[0]\| = 0.0000` |
| 3 | one verify pass == K+1 sequential decode steps, at 7 anchors x 4 acceptance scripts x 2 prompts | 162 rows, **0 hard mismatches**, 1 near-tie |

Gate 2 is the decisive one: **compared against a decode at the same batch width,
the verify pass is bitwise identical.** All observed deviation is the batch
shape, none of it the verify pass.

Gate 3 is freshly anchored at each measurement rather than free-running,
because the KV cache is *written* by these passes as well as read: a batch-1
decode walk and a batch-5 verify walk lay down slightly different K/V and drift
apart over dozens of rounds. An earlier free-running version of this gate
failed for exactly that reason and was measuring drift, not correctness.

**Phase 6 will hit this.** Its planned exit criterion — byte-identical greedy
output with speculation on and off — is not achievable in bf16 once Phase 5
makes speculation actually change batch shapes. (Phase 3's gate passed only
because the draft was loaded and never run, leaving shapes identical.) Either
run that gate in float32 or restate it as distributional equivalence plus
near-tie classification, as gate 3 does here.

## Blocker found: decode is nondeterministic at batch width >= 6

**Pre-existing, unrelated to speculative decoding, reproduces on a clean
checkout with everything in this phase stashed.** Reproduction:

```
python experiments/decode_determinism_check.py
```

Calling the *same* decode step twice on the *same* requests returns different
logits. The pass is idempotent, so they must agree bitwise. Below width 6 they
do; at and above it they diverge by up to 21.7, and at some widths the argmax
changes, i.e. the engine emits a different token. Identical requests batched
together disagree with each other by the same margin.

Ruled out by direct experiment: the `num_splits` heuristic (pinning it to 1
only moves which widths break), `@torch.compile` on `MLP.forward`
(`TORCHDYNAMO_DISABLE=1`, same), and uninitialized KV cache (`zeros` and a
constant fill, same). The kernel reproduces it standalone on random tensors
with no engine involved, which localises it to
`csrc/mfa/decode.cuh`. The signature — clean at low occupancy, corrupting
unpredictably as more blocks become resident — is a missing or mis-scoped
shared-memory barrier. Note the kernel aliases several buffers over one
`extern __shared__` region (`decode.cuh:588`, `:628`) while `flash.cu` sizes
that allocation for Q + K + V only.

Consequences:

- **`max_num_batched_seqs` above ~5 is not currently safe.** The 4 GB profile
  uses 8; the stock `Config` uses 512; `benchmark/` runs far higher.
- Existing gates pass because they run two prompts — batch width 2.
- **It caps K.** A verify pass is width `num_requests * (K+1)`, so one request
  at K=4 is width 5, the last safe width. `verify_pass_gate.py` is gated at
  K=4 for this reason and fails at K=8 with mismatches that belong to this bug.
- Phase 7's throughput sweep over K is blocked on this, not just on Phase 1.

Fixing it means patching `mini-flash-attention` and rebuilding (recipe in
Environment State). That is a separate piece of work and was left out of
Phase 4 deliberately.

## Environment State (BUILT — verified 2026-08-03)

**The engine runs on this host.** `mini-flash-attention` and `triton` are
installed and exercised on GPU; the full runbook steps 0-4 have passed.

### The build recipe, exactly

The one thing that matters: **CUDA Toolkit 12.6, not 13.x.** torch here is
`2.9.1+cu126`, and `torch/utils/cpp_extension.py::_check_cuda_version` raises
`RuntimeError` outright on a CUDA *major* version mismatch. Switching torch to
cu130 instead is not an escape either — the driver is 566.07, and CUDA 13
binaries need r580+.

A CUDA 13.3 toolkit is also installed on this host and is simply unused; the
two coexist without conflict. Point `CUDA_HOME` at v12.6 and ignore 13.3.

1. **MSVC toolset must be pinned to 14.44.** CUDA 12.6's
   `include/crt/host_config.h:168` rejects `_MSC_VER >= 1950`, and the default
   compiler here is VS 2026's 14.51 (`cl` 19.51). Toolset 14.44 (`cl` 19.44) is
   installed alongside it. Select it with:
   `vcvarsall.bat x64 -vcvars_ver=14.44`, and set `DISTUTILS_USE_SDK=1` so
   setuptools honours that environment instead of probing for VS itself.
2. **`mini-flash-attention` needs three Windows fixes in its `setup.py`**
   (upstream is Linux-only): `-std=c++20 -O3` are GCC spellings that MSVC
   ignores with a D9002 warning and must become `/std:c++20 /O2`; and
   `{CUDA_HOME}/lib64` does not exist on Windows, where import libraries live
   in `lib/x64`. Its `include/cccl` entry is also absent in 12.6 (that layout
   arrived in 12.8) but is harmless, since libcu++ sits directly under
   `include/` there.
3. **Clone it to a short path.** The CUTLASS submodule has filenames that blow
   past Windows `MAX_PATH` from a deep directory; the checkout fails with
   "Filename too long". `C:\Users\jerry\mfa-build` works. Only
   `3rd/cutlass/include` is actually referenced, so a sparse checkout of
   `include` at the pinned submodule SHA is enough and much faster.
4. Set `TORCH_CUDA_ARCH_LIST=8.6` so the build targets only this GPU.
5. Install with `pip install --no-build-isolation .` — isolation would build in
   a fresh env without torch, and `setup.py` imports torch at module scope.

The working build script is kept at `C:\Users\jerry\mfa-build\build_win.bat`.

### Windows runtime gotcha: CUDA graphs

`TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` is **required** for
`use_cuda_graph=True`. Without it, graph capture dies with
`OverflowError: Python int too large to convert to C long` inside
`torch/_inductor/runtime/static_cuda_launcher.py` — the static launcher passes
a 64-bit device pointer through a C `long`, which is 32-bit on Windows (LLP64).
`experiments/engine_spec_gate.py` sets it before importing torch. With it set,
graphs capture fine and decode output is bit-identical to eager.

`flashinfer` remains uninstalled and is still not needed: its import is lazy and
greedy decoding takes `Sampler`'s `argmax` branch.

- `.venv/` has `transformers==4.57.3`, tokenizers, safetensors, numpy, xxhash,
  huggingface_hub, plus `mini_flash_attention==0.1.0` (built from source) and
  `triton-windows==3.5.1.post24`. Python is 3.13.3.
- **triton version is not free.** torch 2.9 pairs with triton 3.5.x; `triton-windows`
  publishes up to 3.7.x, and taking the latest risks breaking the
  `@torch.compile` on `MLP.forward`. Pin `3.5.1.post24`.
- **torch: migrated and working.** `2.9.1+cu126`, `torch.cuda.is_available()`
  is **True** (confirmed 2026-08-03). The venv previously held `2.9.1+cu130`,
  which driver 566.07 cannot run. Note `requirements.txt` still pins
  `torch==2.9.1+cu130`; following it verbatim would re-break CUDA on this host.
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

### Windows build risk — RESOLVED

Every item that was listed here as a risk has been retired. `mini-flash-attention`
compiles on MSVC and its kernel was checked numerically against
`torch.nn.functional.scaled_dot_product_attention` (max abs error 9.2e-4 on an
fp16 GQA case, 8 query heads / 4 KV heads, causal). The build recipe is above.

### Running on 4 GB — required tuning

The stock `Config` **cannot start on this GPU**. Measured numbers:

- 4096 MiB total, of which ~2008 MiB is already gone before the cache is sized
  (bf16 weights ~1174 MiB, plus ~834 MiB of CUDA context and Windows desktop).
- The warmup pass peaks at ~1188 MiB.
- At the stock `kv_cache_block_size=256`, one block costs **28 MiB**
  (`2 * 28 layers * 256 * 8 kv heads * 128 head dim * 2 bytes`).

So `gpu_memory_utilization=0.5` leaves a 26 MiB budget against a 28 MiB block
and trips `assert kv_cache_num_blocks > 0`. That assert now carries a message
reporting the budget, the amount already used, the warmup peak, and the
per-block cost, instead of failing bare.

The working local profile — used by `experiments/engine_spec_gate.py`:

| setting | stock | local |
|---|---|---|
| `kv_cache_block_size` | 256 | **64** (7 MiB/block) |
| `gpu_memory_utilization` | 0.5 | **0.9** |
| `max_model_len` | 4096 | **1024** |
| `max_num_batched_tokens` | 16384 | **2048** |
| `max_num_batched_seqs` | 512 | **8** |

Dropping the block size matters more than raising utilization: 28 MiB is far
too coarse a quantum when the whole budget is ~1.6 GB.

## What Has Been Confirmed

- `Executor._build_decode_input` decodes one token per request per step.
- `FlashAttention.forward` stores KV entries during decode and calls
  `flash_attn_with_kvcache(q.unsqueeze(1), ...)`.
- `Sampler` only exposes the existing greedy / top-k / top-p path. Greedy is
  reached with `temperature=1.0, top_k=0, top_p=1.0`, which leaves all three
  tensors `None` and takes the `argmax` branch without importing flashinfer.
- `Config` **does** now contain the speculative-decoding fields (Phase 3).
- `flash_attn_with_kvcache` asserts `seqlen_q == 1` — confirmed, unchanged.
- **`flash_attn_varlen_func` has no `seqlen_q` restriction, accepts a paged
  `block_table`, and supports GQA.** This is the verify path.
- **Latent bug — fixed in Phase 4, and it was worse than recorded.**
  `attention.py` never passed `block_table=` to `flash_attn_varlen_func`. It
  now does, but that alone would not have made prefix caching correct: the
  kernel's causal mask is top-left anchored, so a prefix-cached prefill
  (`seqlen_q < seqlen_k`) attends to the wrong keys regardless. The call site
  now asserts `seqlen_q == seqlen_k` so the path fails loudly.
- **`kv_cache_block_size` must be a multiple of 64** — both kernels resolve one
  block-table entry per 64-key tile. Asserted in `Config` since Phase 4.
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

**None.** The environment blocker that gated Phases 3-6 is cleared: the engine
imports, runs, and passes its correctness gate on this host. No blockers on the
design or the algorithm either.

# RESUME HERE

The environment is built and Phases 0, 2, 3 and 4 are done. **Next work is
Phase 5: rejection sampling in the engine plus bookkeeping.**

### Step 0 — confirm the stack still works (5 minutes)

```
python experiments/engine_spec_gate.py --compare --cuda-graph   # GATE: PASS - byte-identical
python experiments/verify_pass_gate.py                          # GATE: PASS
```

The first exercises both models, the paged cache, prefill, decode, CUDA graphs
and greedy sampling. The second exercises the Phase 4 verify pass. If either
fails, fix it before touching Phase 5 — a broken backend misread as a
rejection-sampling bug is the exact trap this phase ordering exists to avoid.

If the environment needs rebuilding, the recipe is in Environment State above;
the load-bearing detail is **CUDA 12.6, MSVC toolset 14.44**.

### Step 1 — Phase 5

The pieces are all in place: `Executor.verify()` returns
`(num_requests, K+1, vocab_size)` logits, and `rejection_sample()` in
`experiments/spec_decode_prototype.py` was written as a pure function
specifically to be ported unchanged. In order:

1. Port `rejection_sample` into `Sampler` as a **new** method. Do not touch the
   existing `forward` — plain decode uses it.
2. Add a draft-proposal loop. The draft shares the target's block ids
   (Phase 3), so it can reuse `_build_decode_input` against
   `self.draft_kv_cache`; it needs K sequential single-token steps.
3. Wire the round into `Engine.step` / `Executor`: propose -> verify ->
   rejection-sample -> append `num_accepted + 1` tokens. Note
   `Scheduler.update` currently appends exactly one token per request and
   `Metrics` assumes the same; both need to take a count.
4. Call `allocate_block_for_decode(req, extra_tokens=K)` before each verify
   pass — the slots for the proposals must exist before the pass writes them.

Two constraints to respect, both measured:

- **Keep `num_requests * (K+1) <= 5`** until the decode race is fixed. With one
  request that means K <= 4. See the blocker section above.
- **Do not expect byte-identical output** against non-speculative greedy once
  speculation changes batch shapes; see "Why the gate is not byte-identical
  greedy output". Phase 6's exit criterion needs restating.

## Phase Checklist

- Phase 0: **complete — go decision recorded.** Note its identified verify path
  was wrong; corrected in Phase 4. Committed as `8f16197` on branch `test`.
- Phase 1: deferred by design (see above); random-init draft used meanwhile.
- Phase 2: **complete — both gates passing**, 18/18 exact match. See results above.
- Phase 3: **complete and validated on GPU.** Config fields,
  `load_draft_model`, dual KV-cache allocation, draft checkpoint. Exit
  criterion passed byte-identically; see Phase 3 Validation.
- Phase 4: **complete and gated on GPU.** `Executor.verify()` routed through
  `flash_attn_with_kvcache` with K+1 tokens in the batch dimension — *not*
  through `flash_attn_varlen_func`, which cannot express the shape. See Phase 4
  Results.
- Phase 5: **next.** Port `rejection_sample` from the Phase 2 prototype; see
  Step 1 above.
- Phase 6: not started. **Its exit criterion needs restating** — byte-identical
  greedy output is not achievable in bf16 across differing batch shapes.
- Phase 7: needs a trained draft model (Phase 1) *and* the decode race fixed,
  since the sweep varies K and therefore batch width.

## Handoff Notes for Agents

- **Start at "RESUME HERE" above.** The environment is built; next work is
  Phase 4.
- Read [speculative-decoding-plan.md](speculative-decoding-plan.md) first.
- [experiments/engine_spec_gate.py](experiments/engine_spec_gate.py) is the
  engine-level correctness gate and the fastest way to confirm the stack is
  healthy. It carries the 4 GB config profile; the stock `Config` will not
  start on this GPU.
- [docs/spec_decoding_feasibility.md](docs/spec_decoding_feasibility.md) is the
  authoritative note; it records the go decision and the two remaining
  empirical checks on paged-varlen semantics.
- The greedy exact-match harness is the correctness gate at every level —
  prototype (Phase 2, done) and engine (Phase 6, pending).
- Do not modify paging or scheduler core code unless the plan is explicitly revised.
