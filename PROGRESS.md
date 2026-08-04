# Speculative Decoding Progress

Last updated: 2026-08-04

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
- **Phase 1 (draft model training) was deliberately deferred, and is now
  OPTIONAL rather than next.** Correctness never needed it; acceptance rate is
  all it buys — and the **n-gram proposer built instead of it already delivers
  a measured speedup** (~1.7x on repetitive text). If Phase 1 is done later its
  objective is respecified: **distil from the target**, do not pretrain on a
  general corpus (RESUME HERE step 2).
- **Phase 7: complete (2026-08-04).** `benchmark/spec_sweep.py` +
  `docs/spec_sweep.svg` + the README write-up. **1.81x on repetitive text at
  K=4.** Its output-hash gate also turned up the batch-width ceiling on
  byte-identical output — see Phase 7 Results, which is the most interesting
  thing in this file.
- **N-gram / prompt-lookup proposer: built, measured and gated (2026-08-04).**
  A second proposer that needs no model at all. Break-even drops from ~60% to
  ~30% acceptance, and on prompts whose answer copies the question it reaches
  **~1.7x end to end**. On open-ended text it is a ~10% loss, which is the
  honest caveat and is why a round is skipped outright when nothing matches.
  See the N-gram Proposer section.
- **Phase 3: complete and validated on GPU (2026-08-03).** The environment is
  built, the engine runs, and the exit criterion passes byte-identically. See
  Phase 3 Validation.
- **Phase 4: complete and gated on GPU (2026-08-03).** The multi-token verify
  pass exists (`Executor.verify`) and provably equals K+1 sequential decode
  steps. It is **not** built on `flash_attn_varlen_func` as the plan and the
  Phase 0 note assumed — that kernel's causal mask is top-left anchored, which
  makes the shape unusable. See Phase 4 Results.
- **Two pre-existing bugs found along the way, both now addressed.**
  Prefix-caching prefill was doubly broken (missing `block_table` *and* the
  wrong mask anchoring), and **plain decode was nondeterministic at batch width
  >= 6** — a shared-memory race in the backend's decode kernel, unrelated to
  speculative decoding. The latter is root-caused and fixed; see the backend-bug
  section. The fix is a required patch, kept in `patches/`.
- **Phase 5: complete and gated on GPU (2026-08-04).** The engine decodes
  speculatively end to end — draft proposes, target verifies, rejection
  sampling commits between 1 and K+1 tokens per request. Greedy output is
  byte-identical to non-speculative decoding at K=4 and K=8, at both 0% and
  100% acceptance. See Phase 5 Results.
- **Phase 6: complete (2026-08-04).** The correctness harness is a committed
  pytest suite, `tests/test_spec_decode_correctness.py` +
  `tests/test_spec_decode_end_to_end.py`, and it is mutation-checked. See
  Phase 6 Results.
- **Break-even measured, then fixed (2026-08-04).** Speculation was 10-21x
  slower per step than a CUDA-graph decode, making break-even unreachable at
  *any* acceptance rate — the cause was kernel-launch overhead, not draft
  quality. Graphs now cover the verify and propose passes: a K=4 round went
  134.9 ms -> **19.9 ms**, and break-even from impossible to **~60%** (~50% at
  K=2). Acceptance rate then became the binding constraint — and the n-gram
  proposer, not Phase 1, is what cleared it. See the two sections on break-even,
  the N-gram Proposer section, and RESUME HERE.

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

**Predicted here, and partly overturned by Phase 5.** This section originally
concluded that byte-identical greedy output with speculation on and off would be
unachievable in bf16 once speculation changed batch shapes, and that Phase 6's
exit criterion had to be rewritten. Measured in Phase 5, the output *is*
byte-identical in every configuration tried. The mechanism described above is
real — batch shape does move logits by up to 0.5 — but changing an emitted token
needs that perturbation to land on a near-tie, which Qwen3-0.6B's greedy argmax
usually is not. So the criterion stands as written, with a documented tolerance
and the near-tie classification kept. See "About Phase 6's exit criterion" under
Phase 5 Results.

## Phase 5 Results (2026-08-04, RTX 3050 4 GB)

```
python experiments/spec_round_gate.py          # K=4 -> GATE: PASS
python experiments/spec_round_gate.py --k 8    # K=8 -> GATE: PASS
```

### The headline

**Greedy output is byte-identical with speculation on and off**, at K=4 and
K=8, for both prompts, at 0% and at 100% acceptance. This is stronger than the
Phase 4 note predicted was achievable — see "About Phase 6's exit criterion"
below, which is now a much narrower caveat than it was.

| draft | K | acceptance | tokens/request/step | output |
|---|---|---|---|---|
| random-init tiny Qwen3 | 4 | 0.0% | 1.00 | identical |
| random-init tiny Qwen3 | 8 | 0.0% | 1.00 | identical |
| target drafting for itself | 4 | 100.0% | 4.70 | identical |
| target drafting for itself | 8 | 99.0% | 7.83 | identical |

The two drafts are chosen to sit at opposite extremes, because they exercise
different code. The random draft is rejected at i=0 in every round, so every
round takes the residual-resample path and abandons the entire proposal region
as stale. The self-draft accepts every proposal, which is the **only** regime
where the draft finishes a round behind the committed sequence and its catch-up
row does real work; if that row were wrong the draft would propose from a stale
cache and acceptance would collapse after round one. It does not.

The one rejected proposal at K=8 is worth reading rather than filing as noise:
the draft ran at batch width 2 and the target's verify pass at width 9, so at a
near-tie their argmaxes disagreed. Rejection sampling did exactly its job —
rejected, resampled the target's token, and the output stayed identical. That is
the algorithm being indifferent to draft quality, observed directly.

### What Phase 5 added

- `Sampler.rejection_sample(draft_tokens, q, p, generator)` — the Phase 2
  prototype's pure function, batched across requests. `Sampler.forward` is
  untouched; plain decode still runs it. `Sampler.greedy_probs` renders
  temperature 0 as a one-hot distribution so greedy takes the general
  rejection-sampling path rather than a shortcut, as Phase 2 established.
- `Executor.propose(requests) -> (proposals, q)` — K sequential draft steps.
- `Executor.execute_speculative(batch) -> (tokens, num_accepted)` — one round.
  It does not commit; the scheduler does.
- `Executor._build_paged_rows` — `_build_verify_input` refactored onto a shared
  builder that takes an explicit (token, position) list per request, which the
  draft's steps also use. The Phase 4 gate re-passes unchanged against it.
- The draft now prefills alongside the target in `Executor.execute`, over the
  identical `Context` — legitimate because the two caches share block ids. This
  also runs during warmup, which puts the draft's activation peak into the
  numbers `_init_kv_cache` sizes against; Phase 3 had left that unaccounted.
- `Scheduler.update(batch, tokens) -> list[int]` now takes a **list per
  request** and returns how many tokens it actually committed. It stops
  committing at EOS or max_tokens mid-round and drops the rest.
- `Scheduler.decode_extra_tokens` — the decode path allocates `K` tokens of
  slack, and the *preemption check* sees the same number, so a round can never
  be scheduled into a cache that has no room for its proposals.
- `Metrics` counts committed tokens rather than batch width, and tracks
  `acceptance_rate` and `tokens_per_request_step`. The latter is deliberately
  per request, not per batch step, or the batch width would be folded into the
  number Phase 7 is trying to chart.
- CUDA graphs are now *skipped* rather than captured-and-unused when
  speculation is on. Every graph is captured for one query row per request and
  a round runs K+1, so none could ever be replayed.

### Two bugs fixed on the way

- `cache_block_if_needed` indexed `req.blocks[-1]`, which is the block holding
  the last *token* only when a request holds exactly `cdiv(len, block_size)`
  blocks. A speculative round reserves K tokens of slack, so it can hold
  trailing empty blocks and the hash would have landed on the wrong one. Now
  indexed by the last token's position — provably identical in every
  non-speculative case.
- The gate script's round recorder held a *bound* method of the executor, which
  pinned the model weights and KV cache of every engine it had wrapped. On a
  4 GB card the third engine then had no memory to size a cache against. Worth
  recording because the symptom — "not enough VRAM" — pointed nowhere near it.

### Scope: greedy only, deliberately

`execute_speculative` asserts `top_k <= 0 and top_p >= 1.0`. Rejection sampling
must compare the *exact* distribution the non-speculative path would have
sampled from, and for top-k/top-p that distribution is constructed inside
flashinfer's fused kernel, which never exposes it. Reconstructing it outside
would be a second implementation to keep in sync, and every correctness gate in
this project is greedy. Note temperature needs no special handling: with top-k
and top-p unset, `Sampler.forward` takes its argmax branch whatever the
temperature, because scaling logits cannot move the argmax — so one-hot `p` is
the faithful reproduction of it.

Speculative decoding is also asserted incompatible with prefix caching (off by
default). A multi-token commit can step over the block boundary that
`cache_block_if_needed` hashes on, leaving a gap in the prefix chain.

### Known limits, for Phase 7

- **`p` and `q` are dense over the vocabulary.** A round materialises
  `(B, K+1, V)` and `(B, K, V)` float32 tensors — at B=8, K=8 and Qwen3's 151936
  vocab that is ~85 MiB per round, and it grows linearly in `max_num_batched_seqs`.
  For greedy specifically this is nearly all waste, since one-hot rows carry
  `B*(K+1)` integers of information. It is kept because representing temperature
  0 as a distribution is what makes greedy run the *general* rejection-sampling
  path, which is the property the whole correctness argument rests on (Phase 2).
  If Phase 7 wants wide batches, the fix is a fused kernel that never
  materialises the one-hot, not a greedy special case in the sampler.
- Nothing here has been run at the stock `max_num_batched_seqs=512`, and the
  point above is the first thing that would break there.
- The draft's proposal loop is K sequential forward passes. It is the obvious
  target once acceptance rate is worth optimising, and it is why a *trained*
  draft (Phase 1) matters: at 0% acceptance those K passes are pure overhead,
  which is exactly what the random-draft row of the table above shows.

### About Phase 6's exit criterion

Phase 4 predicted byte-identical greedy output would be unachievable once
speculation changed batch shapes. Measured, it is achievable here — 4 runs, 2
prompts, K in {4, 8}, both acceptance extremes, all identical. The prediction
was not wrong about the mechanism, only about how often it bites: the batch
shape does move logits by up to 0.5 absolute, but flipping an emitted token
additionally requires landing on a near-tie, and Qwen3-0.6B's greedy argmax is
usually not that close. So the criterion is **usable as written, with a
documented tolerance** rather than as an unconditional guarantee — a longer run,
another model, or another K may well produce a near-tie flip, and that is not a
bug. `spec_round_gate.py`'s gate 2 already classifies exactly this: it saw 2
near-tie reorderings at K=8 and 0 at K=4, none of which changed the output.

## Phase 6 Results (2026-08-04)

```
pytest tests/ -m "not gpu"   # 30 passed  — no CUDA, no checkpoints, ~12 s
pytest tests/                # 38 passed  — adds the one-round GPU tests, ~32 s
pytest tests/ --slow         # 43 passed  — adds the engine comparisons, ~86 s
```

(Counts as of the n-gram proposer landing; the "not gpu" tier grew most, since
the lookup itself is exhaustively testable in pure Python.)

### The suite

Three tiers, cheapest first, so a contributor without a GPU still gets a real
signal and CI can pick a level:

| tier | file | needs | what it pins down |
|---|---|---|---|
| maths | `test_spec_decode_correctness.py` | nothing | the sampler's output distribution equals the target's; `len(tokens) == num_accepted + 1`; the accepted prefix is never rewritten; greedy's degenerate behaviour |
| one round | same file, `gpu` marker | CUDA + target | `verify()` at K=0 matches decode **bitwise**; a round emits exactly `truth[:M+1]` for M swept 0..K at 3 anchors |
| engine | `test_spec_decode_end_to_end.py`, `slow` | CUDA + both models | greedy output identical with speculation on and off, at 0% and ~100% acceptance, with per-round invariants checked |
| proposer | `test_ngram_proposer.py` | nothing | the n-gram lookup, exhaustively: longest match wins, most recent among equals, the quality floor, padding to K, and that the history is never mutated |

`tests/spec_harness.py` holds the primitives. The dependency deliberately runs
**tests → experiments**, not the reverse: `experiments/verify_pass_gate.py` and
`experiments/spec_round_gate.py` now import from the harness rather than
defining their own copies. A regression suite that breaks when someone edits an
exploratory script is a regression suite nobody trusts.

### Mutation-checked, because a green suite proves nothing on its own

Two deliberate bugs were introduced and the suite was re-run:

| mutation | caught by |
|---|---|
| `Sampler.rejection_sample` accepts every proposal | 7 tests — both distribution tests, the greedy degenerate test, and the round test at M=0..3 |
| `_build_paged_rows` off by one in `cache_seqlens` | 6 tests — the bitwise `verify()` test and the round test at every M |

Worth noting the first mutation left `M=K` passing, which is correct: when every
proposal *should* be accepted, a sampler that always accepts is right by
accident. The parametrisation is discriminating rather than uniformly loud.

### tests/test_block_manager.py was failing at HEAD, and is fixed

Unrelated to speculative decoding. It asserts on `hash_to_block_id` and
`num_cached_tokens`, which only move when prefix caching is on, but constructed
`KVCacheBlockManager` without `support_prefix_cache=True` — the default flipped
to `False` at some point and left the test red. Fixed by asking for the flag the
test was always testing. This mattered beyond tidiness: while the suite was red,
"the tests pass" was not a statement anyone could make, and a new regression test
added to a red suite is a regression test nobody reads.

### Root conftest.py

New, and load-bearing rather than decorative. pytest puts the directory holding
the rootmost `conftest.py` on `sys.path`, which is what makes `import minivllm`
and `import tests.spec_harness` resolve under any invocation. Without it,
`pytest tests/` and `python -m pytest tests/` differ — the latter worked only
because it happens to prepend the working directory. Both are now verified, from
inside the repo and from outside it. It also registers the markers and the
`--slow` option, and sets `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` before
torch is imported.

## Phase 7 Results (2026-08-04) — the sweep, the chart, and one real finding

`benchmark/spec_sweep.py` runs every proposer against K on two workloads, one
configuration per process, and writes `docs/spec_sweep.svg` plus
`benchmark/results/spec_sweep.json`. It also hashes each configuration's output
tokens and compares them against plain decoding's, so the chart is a correctness
gate as well as a benchmark.

### The numbers (K=4 unless stated; 2 concurrent requests, graphs on)

| proposer | workload | tok/s | vs plain | acceptance |
|---|---|---|---|---|
| plain decode | repetitive | 135 | 1.00x | — |
| **n-gram** | repetitive | **244** | **1.81x** | 81% |
| plain decode | open-ended | 200 | 1.00x | — |
| n-gram | open-ended | 189 | 0.94x | 18% |
| draft model, random init | open-ended | 94 | 0.47x | 0% |
| draft model, self-draft | open-ended | 170 | 0.85x | 97% |

The last two rows are the argument for why Phase 1 was not the cheap route to a
speedup, and they are worth reading together. The random draft is the *cost*
curve at the intended 42M size: 0.47x before any proposal quality exists. The
self-draft is the *acceptance ceiling*: 97% acceptance, and still 0.85x, because
a 0.6B draft is not cheap. A trained draft lives between them. See
[docs/future_upgrades.md](docs/future_upgrades.md) for the full estimate.

n-gram at K=1..8 on repetitive text: 1.28x, 1.44x, **1.81x**, 1.77x. The peak at
K=4 is the tradeoff the plan predicted — too small a K wastes the opportunity,
too large a K spends verify capacity on proposals that die after the first
divergence.

### The finding: byte-identical output has a measured batch-width ceiling

The sweep's hash check fired on all three K=8 open-ended configurations. It was
not a fluke and not a near-tie, and running it down produced the most useful
result of the phase.

**What it is not.** Not the decode-kernel race (`decode_determinism_check`
passes at every width). Not CUDA graphs (reproduces eager). Not the K-token block
reservation (a K=8 config whose proposer *never* fires is byte-identical). Not
accumulated run length (K=4 is identical at 256 tokens; K=8 diverges at token 53
regardless of budget). Not the round logic — the divergent step was itself a
plain-decode fallback.

**What it is.** A verify pass over `num_requests * (K+1)` rows is a wider GEMM
than a decode step, and past a certain width cuBLAS retiles and the QKV
projection writes **different K/V into the paged cache for the same token**.
`experiments/kv_shape_drift.py` measures it:

| rows/request | K | batch width | max abs dK vs decode | max abs dV |
|---|---|---|---|---|
| 1 | 0 | 2 | 0.000000 | 0.000000 |
| 5 | 4 | 10 | 0.000000 | 0.000000 |
| 7 | 6 | 14 | 0.000000 | 0.000000 |
| **9** | **8** | **18** | **1.000000** | **1.125000** |
| 17 | 16 | 34 | 4.000000 | 0.843750 |

**Why this matters more than the logit noise already documented.** Logit noise is
transient — the argmax may flip at a genuine tie and nothing persists. KV noise
is *permanent*: both runs keep decoding but no longer over the same numbers, so
they can part company arbitrarily far downstream, at a step where plain decode
was completely certain. That is exactly what happened at token 53 with a top-2
gap of 14.6.

**The output is still correct.** Rejection sampling stays exactly
distribution-preserving, so a wide round still emits a valid sample from the
target — just not the same sample plain decode would have produced. Byte
identity is a reproducibility property, not a correctness one, and above width 14
it is unachievable at any acceptance rate.

**The practical rule.** Byte-identical greedy output holds while
`num_requests * (K+1) <= 14` on this host. At 2 concurrent requests that means
K <= 6. The gate in `spec_sweep.py` splits mismatches accordingly rather than
widening until everything passes, and `experiments/spec_divergence.py` classifies
a divergence as tie / KV-drift / bug with the width checked *first*.

### A methodological correction worth keeping

The first version of `spec_divergence.py` classified the K=8 divergence as a hard
mismatch on a 14.75 top-2 gap. That verdict was wrong twice over. It measured the
gap by prefilling `prompt + tokens[:at]` and taking one step — a *third* cache
state neither run was ever in, since a prefill and a decode walk write different
K/V for the same tokens. And even measured correctly, the gap test only
distinguishes "tie" from "bug" while both runs share a cache, which above the
drift width they do not. Both traps are now documented in the script.

## N-gram Proposer (2026-08-04) — built, measured, gated

**The alternative to Phase 1, and it landed.** `speculative_method="ngram"`
replaces the draft *model* with a search over the request's own token history:
match the last few committed tokens against earlier text, propose whatever
followed them that time. No parameters, no second forward pass, no second KV
cache. It is a production technique — vLLM ships it as `[ngram]`, HF
transformers as `prompt_lookup_num_tokens`.

### Why it needed no changes to the verify-and-accept core

A lookup is a *deterministic* proposer, so its proposal distribution `q` is
one-hot on the token proposed — exactly what `Sampler.greedy_probs` returns for
a greedy draft model. `Executor.verify`, `Sampler.rejection_sample`, the
scheduler's multi-token commit, `Scheduler.decode_extra_tokens` and the whole
Phase 6 argument are untouched and carry over verbatim. **A proposer changes the
acceptance rate; it can never change a token.**

### What it costs, and what that does to break-even

`experiments/spec_breakeven.py --method ngram`, batch 1, context 256, graphs on:

| proposer | propose | round (K=4) | round/T_decode | break-even acceptance |
|---|---|---|---|---|
| draft model | 7.3 ms | 19.9 ms | 2.24 | ~60% |
| **n-gram** | **0.25 ms** | **12.6 ms** | **1.41** | **~30%** |

Flat in K, as the draft path is not: 11.8 ms at K=1, 12.8 ms at K=8. The round
is now almost entirely one `verify()` pass (~9.7 ms of 12.6), which is itself
within ~10% of a plain decode step because launch overhead dominates both.

It also hands back the draft's memory: **237 KV-cache blocks against 185** at
the local 4 GB profile.

### End-to-end, on two workloads — and both belong in the write-up

`--end-to-end ngram_graph --workload {open,repetitive}`, K=4, 128 tokens,
2 requests, medians of paired runs against a graphed non-speculative baseline:

| workload | baseline | n-gram | speedup | acceptance | rounds run |
|---|---|---|---|---|---|
| repetitive (answer copies the question) | 138 tok/s | **234 tok/s** | **~1.7x** | 79% | 44% of steps |
| open-ended (chat) | 209 tok/s | 192 tok/s | ~0.9x | 18% | 4% of steps |

Run-to-run spread on this laptop is several percent; do not quote these to a
decimal. **The second row is not a footnote.** N-gram lookup wins on repetitive
workloads — summarisation, code editing, RAG, "rewrite this paragraph" — and
does nothing on open-ended generation, where the tail has simply never occurred
before. Reporting only the first row would be dishonest, and reporting both is
what makes this a result rather than a demo.

### The skip path is what bounds the downside

When no request in a batch has a match, `propose_ngram` returns `(None, None)`
and `execute_speculative` falls back to `executor.execute(batch)` — a plain
decode step. Without it, every open-ended step would pay 1.41x for a token it
could have had for 1.0x, and the second row above would read ~0.7x instead of
~0.9x. `Metrics.speculation_rate` reports how often a round actually ran, and it
should always be read next to `acceptance_rate`: a high acceptance rate over 4%
of steps is a different claim from the same rate over all of them.

### `ngram_min_match_len` is a quality floor, and its default is measured

A 1-token match exists nearly everywhere in text — every repeated comma is one —
so a low floor makes the proposer speculate on essentially every step regardless
of evidence, and a round costs ~1.4x a decode step. Measured across both
workloads, **3 beat 2 on both**: it roughly halves how often the proposer fires
while *raising* acceptance, because the rounds it drops are the losing ones.
Default is 3. Raising it further keeps shrinking the open-ended loss toward
zero (min=5 gives 194 tok/s at 0.7% speculation) at the cost of the repetitive
win, so it is a real dial and Phase 7 should sweep it.

### Gates

- `pytest tests/test_ngram_proposer.py` — 22 exhaustive unit tests of the
  lookup itself. Pure Python, no GPU, no checkpoints, runs in 0.1 s.
  Mutation-checked against five ways of getting it wrong (earliest-match-wins,
  shortest-pattern-first, no padding, self-match allowed, floor ignored); all
  five caught.
- `pytest tests/test_spec_decode_correctness.py -k ngram` — `q` really is
  one-hot and matches the proposals; a no-match batch declines.
- `pytest tests/test_spec_decode_end_to_end.py --slow -k ngram` — byte-identical
  greedy output on *both* workloads, plus assertions that the proposer actually
  fired and actually landed. The open-ended case is not redundant: it is
  dominated by the skip path, which is a second route through
  `execute_speculative` that the round-shaped assertions would not otherwise
  cover.
- Full suite: `pytest tests/ --slow` -> **43 passed**. `spec_round_gate`,
  `verify_pass_gate` and `engine_spec_gate --compare --cuda-graph` all still
  PASS.

### One interface change

`Executor.execute_speculative` returns `(tokens, num_accepted, num_proposed)`
rather than a pair. `num_proposed` is 0 for a skipped step, and it is returned
rather than recomputed by the caller so the acceptance rate stays honest —
folding a phantom K proposals into the denominator would understate the proposer
exactly when it is being correctly conservative. `Engine.step` and both
`RoundRecorder`s were updated.

## CUDA graphs for the speculative path (2026-08-04) — FIXED

The blocker below is fixed. A K=4 round went from **134.9 ms to 19.9 ms** and
break-even from *impossible* to **58% acceptance**.

Batch 1, 256 tokens of context, RTX 3050, graphs on everywhere:

| K | round | propose | verify | sample | round/T_decode | acceptance needed | ceiling |
|---|---|---|---|---|---|---|---|
| 1 | 13.6-13.8 ms | 1.7 | 9.3 | 1.9 | 1.55-1.59 | ~57% | ~1.3x |
| 2 | 15.2-15.3 ms | 3.3 | 9.4 | 2.1 | 1.72-1.77 | **~50%** | ~1.7x |
| 4 | 19.9-20.3 ms | 7.1 | 10.1 | 2.2 | 2.24-2.37 | ~60% | ~2.2x |
| 8 | 27.5-28.2 ms | 16.4 | 11.6 | 2.7 | 3.10-3.29 | ~70% | ~2.8x |

Plain decode is 8.6-8.9 ms in the same engine. Ranges are two runs, not error
bars — treat these as ±3 points on the acceptance figures, which is why they are
stated to the nearest few percent rather than to a decimal. Every number in a
given run comes from one process, so the ratios — which is all break-even
depends on — carry no cross-engine contamination. Cross-checked end to end:
131.8 ms/step eager speculation against **23.1 ms/step** graphed, with a
9.1 ms/step baseline.

Per-component speedups: `verify()` 8.1x, `propose()` 7.5x, whole round 6.8x.

### What changed

- `Executor.__init__` sizes the target's runner at
  `max_num_batched_seqs * (K+1)` when speculation is on, and builds a **second
  `CudaGraphRunner` for the draft** at `2 * max_num_batched_seqs` (the draft
  contributes one row per request per step, two on the first).
- `verify()` and `propose()` route through `Executor._decode_forward`, which
  replays a captured graph when one fits and falls back to eager otherwise —
  so graph sizing is a performance question, never a correctness one.
- Phase 5's "skip capture under speculation" is reverted. It was right that no
  graph captured for one row per *request* fits a verify pass; it was wrong that
  this made graphs inapplicable. The constraint is one row per *batch entry*,
  and Phase 4 built the verify pass to satisfy exactly that.

Verified replaying at the expected widths rather than silently falling back:
with 2 requests at K=4 the target replays at width 10 and the draft at 4 (step
0) and 2 (steps 1-3).

### Two latent graph bugs fixed on the way, both reachable from plain decode

- `batch_size_list` was `[1, 2, 4, 8] + range(16, max+1, 16)`, so the largest
  captured size was `max_num_batched_seqs` only when that happened to be 1/2/4/8
  or a multiple of 16. At `max_num_batched_seqs=20`, `replay()`'s
  `next(b for b in batch_size_list if b >= bs)` raised `StopIteration` on a full
  decode batch.
- The small sizes were unconditional, so `max_num_batched_seqs < 8` captured a
  graph wider than the buffers backing it and silently truncated.

### What to aim at now

**K=2 has the lowest break-even (~50%), not K=4.** `verify()` plus the sampler
is a fixed ~11.6 ms floor and `propose()` adds ~1.8 ms per step, so small K
spreads the floor over too few tokens while large K pays too much drafting.
K=2-4 is the band worth training a draft against; K=8 buys a higher ceiling only
if acceptance is very high.

`rejection_sample()` is now 8-15% of a round (2.1-2.8 ms), up from ~2% when
everything else was slow. It is all dense one-hot tensors over a 151936-token
vocabulary plus `.tolist()` syncs — see "Known limits" under Phase 5. That is
the next optimisation if one is wanted, worth roughly 10% of a round.

## Break-even measurement (2026-08-04) — the finding that prompted the above

```
python experiments/spec_breakeven.py
for m in base_eager base_graph spec; do
    python experiments/spec_breakeven.py --end-to-end $m    # cross-check
done
```

**Finding: speculation cannot pay off on this host until the speculative path
can use CUDA graphs, and no amount of draft training changes that.** Kept as the
record of why the graph work happened; the numbers below are the *before*.

Batch 1, 256 tokens of context, RTX 3050:

| | ms/step | vs graphed decode |
|---|---|---|
| plain decode, CUDA graphs | **8.7** | 1.0x |
| plain decode, eager | 68.6 | 7.9x |
| speculative round, K=1 | 92.7 | 10.6x |
| speculative round, K=4 | 134.9 | 15.4x |
| speculative round, K=8 | 188.1 | 21.5x |

Break-even, against the graph-accelerated baseline that production decode
actually runs:

| K | round / T_graph | E[M] needed | acceptance needed | ceiling at 100% acceptance |
|---|---|---|---|---|
| 1 | 10.6 | 9.6 | **impossible** | 0.19x |
| 2 | 12.1 | 11.1 | **impossible** | 0.25x |
| 4 | 15.4 | 14.4 | **impossible** | 0.32x |
| 8 | 21.5 | 20.5 | **impossible** | 0.42x |

"Impossible" is not rhetorical: break-even needs E[M] larger than K, so even a
draft that is accepted every single time leaves speculation 2.4x-5x *slower*
than the baseline. Confirmed end to end — 191 tok/s graphed baseline against
14 tok/s speculative.

### The cause is launch overhead, not the draft

CUDA graphs are worth **7.9x** on plain decode here. That is the whole story: at
batch 1 a 0.6B model in bf16 is memory-bandwidth-bound with a floor near 6 ms,
and the graphed step hits 8.7 ms while the eager one spends ~60 ms launching
kernels. Windows WDDM launch cost on a laptop GPU sharing a display is brutal.

Phase 5 disabled graph capture under speculation because no captured graph can
match a K+1-row verify pass. Measured, that decision — correct in itself — is
what makes the feature unusable, not draft quality:

- `verify()` costs ~80 ms and is **flat in K** (80.6 / 81.0 / 80.9 / 75.2 ms at
  K=1/2/4/8). Exactly as the technique predicts: K+1 rows are nearly free once
  the weights are resident. That part works.
- `propose()` costs ~13-15 ms **per draft step** and so grows linearly: 13.3 ms
  at K=1 up to 124.7 ms at K=8. A 4-layer/256-hidden draft should be ~1/30 of a
  28-layer/1024 target by FLOPs; it measures 1/5. That gap is launch overhead,
  and it is why K=8 costs more than K=4 rather than amortising better.
- `rejection_sample()` costs 1.8-3.9 ms — small, but ~15% of what an optimised
  round would be, and it is all dense one-hot vocab tensors plus `.tolist()`
  syncs. See "Known limits" under Phase 5.

### The fix, and why it was tractable — DONE, see the section above

Phase 4's central finding was that a verify pass is *decode-shaped*: K+1 tokens
per request ride in the **batch** dimension. So a verify pass at B requests and K
proposals is exactly a decode call at batch width `B*(K+1)` — the same shape
`CudaGraphRunner` already captures, with the same `Context` fields, and
`replay()` already pads up to the next captured size with `slot_mapping = -1`.
The draft's proposal steps are likewise decode-shaped at width B (2B on step 0),
against the draft model.

The prediction recorded here was "a K=4 round lands near 19-20 ms against an
8.7 ms baseline — ratio ~2.2, break-even ~55%, ceiling ~2.2x." Measured after
the work: **19.9 ms, ratio 2.24, break-even ~60%, ceiling ~2.2x.**

### Caveats on these numbers

- Batch 1 and 256 tokens of context. Larger batches shift things toward
  speculation being *worse*, since a wider verify pass costs more while the
  baseline step amortises better.
- This is a 0.6B target. Speculative decoding is aimed at models where one
  forward pass is expensive relative to fixed overheads; 0.6B on a laptop is
  close to the least favourable case, and the technique's reputation comes from
  7B+ models where launch overhead is a rounding error.
- The microbenchmark was cross-checked against `Engine.generate` wall time and
  the ratios agree (see the script's docstring). It is not measuring an artefact.

## Backend bug found and FIXED: decode was nondeterministic at width >= 6

**Pre-existing, unrelated to speculative decoding — it reproduced on a clean
checkout with everything from Phase 4 stashed.** Found while gating Phase 4,
root-caused, fixed, and verified.

### The bug

Calling the *same* decode step twice on the *same* requests returned different
logits. The pass is idempotent, so they must agree bitwise. Below width 6 they
did; at and above it they diverged by up to 21.7, and at some widths the argmax
changed — the engine emitted a different token. Identical requests batched
together disagreed with each other by the same margin.

`flash_attention_fwd_split_kv_kernel` aliases two things over the same
`extern __shared__` region: `warp_max_val` / `warp_expsum_val`, the cross-warp
softmax reduction (`decode.cuh:588`), and `warp_output` (`:628`), both at
offset 0. Every warp reads the reduction values at `:604` and `:613`, then each
warp writes 128 floats of output from offset 0 at `:512`, clobbering them.
**No barrier sat between the reads and the write**, so warp 0's write raced the
other warps' reads.

The corruption lands on the softmax normalisation, which is why the damage was
a *rescaled* output rather than a small perturbation. It only manifested once
occupancy let warps drift out of lockstep — hence the batch-size dependence.

Three wrong suspects were eliminated first, each of which *moved* the failing
widths without fixing anything (exactly what perturbing occupancy does to a
scheduling-sensitive race): the `num_splits` heuristic, `@torch.compile` on
`MLP.forward`, and uninitialized KV cache. `compute-sanitizer --tool racecheck`
then named it outright — ~4,500 hazards per launch between `decode.cuh:512` and
`:604`/`:613`.

### The fix

One `__syncthreads()` between the reduction and the output write, kept at
[patches/mini-flash-attention-decode-race.patch](patches/mini-flash-attention-decode-race.patch)
with application instructions in [patches/README.md](patches/README.md). It must
be applied to any rebuild of the backend; upstream `w4096/mini-flash-attention`
still has the bug.

Verified after rebuilding:

- racecheck: 4,500 hazards -> **0**.
- `decode_determinism_check.py`: deterministic at **every** width 1-18 through
  the engine and 1-32 in the bare kernel, intra-batch spread `0.0000`
  everywhere (was up to 30.5).
- `verify_pass_gate.py`: passes at **K=4, K=8 and K=16** — 162 / 266 / 434 rows,
  0 hard mismatches. K=8 previously failed with 6.
- `engine_spec_gate.py --compare --cuda-graph`: still `byte-identical`.

### Correction to an earlier claim

An earlier revision of this file said "treat width <= 5 as safe". That was
wrong. The race existed at **every** width and merely failed to manifest when
few resident blocks kept the warps in lockstep. There was never a safe ceiling,
only an unobserved one — which also means the Phase 4 gate results collected
before the fix were "not observed to fail" rather than proven. They have since
been re-run against the fixed backend and are clean.

### Still worth knowing

`max_num_batched_seqs` is no longer bounded by this, but nothing in this repo
has yet been *tested* at the stock 512 or at the widths `benchmark/` uses. The
determinism check covers up to 32.

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

## Deviations from the plan, and why

**1. Phase 1 (train the draft model) was deferred until after Phase 6.** Done as
intended; it is now the next work.

Rejection sampling returns the target model's exact distribution regardless of
draft quality — a poor draft lowers the acceptance rate but can never change
the output tokens. So the entire correctness effort (Phases 2, 4, 5, 6) was
driven by a randomly-initialised tiny Qwen3 sharing the target's vocabulary, and
a trained draft is needed only for the speedup.

Phase 2 confirms this empirically: the random-init draft achieved 0% acceptance
and still reproduced greedy output exactly, across all three K values. Phases 5
and 6 confirm it through the engine.

**2. Phase 1's training objective is respecified: distillation from the target,
not pretraining on a general corpus.** The plan says a web-text or
instruction corpus is fine because the draft's usefulness "is measured entirely
by acceptance rate against the target, not by its own perplexity" — which is
true, and is the reason the conclusion does not follow. If acceptance against
the target is the metric, agreement with the target is the objective. See
RESUME HERE step 2 for the recipe and the reasoning. Note Phase 1 is now
optional — see step 1 for the n-gram proposer that reaches a measured speedup
without it.

**3. Phase 4's verify mechanism** is the decode kernel with K+1 tokens in the
batch dimension, not `flash_attn_varlen_func` — see Phase 4 Results.

**4. The draft's KV cache is paged and shares the target's block ids**, rather
than being the separate contiguous tensor the plan calls for — see Phase 3.

## Blockers

**None.** The environment blocker that gated Phases 3-6 is cleared: the engine
imports, runs, and passes its correctness gate on this host. No blockers on the
design or the algorithm either.

# RESUME HERE

**Every phase in the plan is complete.** Phases 0 and 2-7 are done, Phase 1 is
deliberately optional and respecified. Speculative decoding works end to end, is
guarded by a committed test suite, is measurably faster on repetitive workloads
through the n-gram proposer, and is written up with a chart.

Remaining ideas, with the measurement justifying each, are in
[docs/future_upgrades.md](docs/future_upgrades.md).

### Step 0 — confirm the stack still works (10 minutes)

```
pytest tests/ --slow                                            # 43 passed
python experiments/engine_spec_gate.py --compare --cuda-graph   # GATE: PASS - byte-identical
```

The suite is the fastest complete answer; the gate script additionally exercises
CUDA graphs, which the tests leave off. If either fails, fix it before building
on top — a broken backend misread as a rejection-sampling bug is the exact trap
this phase ordering exists to avoid, and
`python experiments/decode_determinism_check.py` settles that question in one
run (it fails loudly if a backend rebuild dropped
`patches/mini-flash-attention-decode-race.patch`).

If the environment needs rebuilding, the recipe is in Environment State above;
the load-bearing detail is **CUDA 12.6, MSVC toolset 14.44**.

### Step 1 — DONE: the n-gram proposer

**Built, measured and gated on 2026-08-04.** Full results in the N-gram Proposer
section above; the short version is `speculative_method="ngram"`, break-even
~30% instead of ~60%, **~1.7x end to end on repetitive text and ~0.9x on
open-ended text**. Phase 1 is optional as a result rather than load-bearing.

Nothing to do here unless the open-ended case is worth attacking, which is what
step 2 is for. Two things to know before touching it:

- **`ngram_min_match_len` is the dial**, and Phase 7 should sweep it. Higher
  means fewer rounds on better evidence; it shrinks the open-ended loss and the
  repetitive win together.
- **The skip path is load-bearing**, not an optimisation. Removing it turns the
  open-ended ~0.9x into ~0.7x.

### Step 2 (OPTIONAL) — Phase 1: train the draft, by DISTILLATION not plain pretraining

Worth doing if you want the general-purpose case — a trained draft helps on
open-ended generation where lookup cannot. No longer on the critical path for
having something to show. Budget: ~1 day of active work plus 2-5 days of
mostly-unattended compute, and **do it in sprints** — the dominant risk is "will
acceptance clear ~50%", which a 10M-token run answers in an afternoon, not "will
it converge".

CUDA graphs for the speculative path are **done** (see the section above), so
acceptance rate is finally the thing that decides whether the feature pays.

**The target to beat: ~50% acceptance at K=2, ~60% at K=4.** Below that,
speculation is slower than plain graphed decode. Above it, the payoff runs up to
a ceiling of ~1.7x / ~2.2x. K=8 needs ~70% to break even at all, so aim at
**K=2-4**.

#### Why the plan's Phase 1 spec would undershoot

`speculative-decoding-plan.md` section 4 says the draft's training data "doesn't
need to match Qwen3's training data — a general web text or instruction-style
corpus is fine," because "the draft model's job is to propose plausible
continuations, not to be a good model — its usefulness is measured entirely by
acceptance rate against the target, not by its own perplexity."

The second half of that sentence is exactly right and the conclusion drawn from
it is exactly wrong. If usefulness is measured by acceptance against the target,
then the training objective should be *agreement with the target*, not
"plausible continuations." Those come apart badly: a 20M model trained on a web
corpus can be a perfectly reasonable little LM and still sit well under 50%
agreement with Qwen3-0.6B's argmax, because it is a different model that learned
a different distribution. Nothing in plain next-token cross-entropy on unrelated
text is aimed at the number that gates this feature.

The literature agrees — this is DistillSpec (Zhou et al. 2023) and, for the
cheap variant, sequence-level knowledge distillation (Kim & Rush 2016).

#### The recipe, ordered by cost

**Start with sequence-level distillation.** Generate a corpus *with
Qwen3-0.6B itself* — prompts from any diverse source, completions sampled from
the target at low temperature — then train the draft with ordinary next-token
cross-entropy on those completions. This is the cheap variant and it matters
disproportionately here:

- **No target model resident during training.** On a 4 GB card that is close to
  decisive: the target's bf16 weights are ~1.17 GB before any activations or
  optimiser state, and holding it alongside a draft plus Adam moments plus
  backward activations is a fight you do not need to pick.
- Generation is a one-off cost, reusable across every training run and every
  architecture you try.
- It is ordinary CE training, so the plan's existing loop needs no change —
  only its *data* changes.

**Upgrade to logit distillation only if acceptance falls short.** Train against
the target's top-k logits with a KL objective rather than against its sampled
token. Strictly more signal per token, and the standard way to squeeze out the
last few points. Two ways to pay for it:

- *Offline*: cache top-k logits during corpus generation. At top-32 that is
  ~192 bytes/token (fp16 value + int32 index), so ~1.9 GB per 10M tokens — chunky
  but it keeps the target out of the training process.
- *Online*: run the target in the training loop. Cleanest and most flexible;
  needs the VRAM headroom this host does not obviously have.

#### Architecture — the plan's guidance stands

10-30M non-embedding parameters, **Qwen3's tokenizer exactly** (`Config` asserts
matching vocab size, and a mismatch would be silent corruption rather than a
crash), tied embeddings so the 151936-token vocab does not dominate the
parameter count. `experiments/make_random_draft.py` already builds a Qwen3 of
this shape and self-checks the tokenizer against the target's — reuse it for the
model definition and add a training loop, rather than starting from scratch.

Note the existing random draft is 42.0M total / **3.1M non-embedding**, which is
below the plan's 10-30M band. It was sized to make the correctness harness fast,
not to be accepted. Expect to grow it.

#### Measure acceptance early and often — the loop already exists

Do not train to convergence before finding out whether it is working. Acceptance
is already instrumented end to end:

```
python experiments/spec_round_gate.py --draft <path>    # prints acceptance + histogram
python experiments/spec_breakeven.py --end-to-end spec_graph
```

`Metrics` tracks `acceptance_rate` and `tokens_per_request_step` live, and
`spec_round_gate.py`'s gate 3 prints the full acceptance histogram, so you can
see whether the draft is being rejected at i=0 every round or getting partway.
Checkpoint early, measure, and only then decide whether to scale the model or
the data. A run that ends at 35% acceptance is a run that should have been
stopped and rethought at 20%.

Also re-run the correctness suite with the new draft — `pytest tests/ --slow`
— since a *good* draft exercises partial-acceptance paths that neither the
random draft (0%) nor the self-draft (~100%) reaches.

### Step 3 — DONE: Phase 7

**Done on 2026-08-04.** `benchmark/spec_sweep.py`, `docs/spec_sweep.svg`, and
the README's Speculative Decoding section. Results in Phase 7 Results above.

What is left, in order of value, is in
[docs/future_upgrades.md](docs/future_upgrades.md) — the distillation corpus
(step 2 below, respecified with measured cost estimates), the batch-width
ceiling on byte identity, top-k/top-p support, and prefix caching.

The original framing for this phase, kept because it is what the chart ended up
showing: include the **break-even line**, not just throughput versus K — the
interesting result on this hardware is *why* speculation does or does not pay,
and that turned out to be a story about launch overhead before it was ever a
story about draft quality.

Chart **two proposers across two workloads** (repetitive and open-ended); step 1
landed, so both axes have real data behind them, and
`experiments/spec_breakeven.py` already produces every number the figure needs:
`--method {draft,ngram}` for the microbenchmark table, `--end-to-end
{base,spec,ngram}_graph --workload {open,repetitive}` for throughput.
`tests/spec_harness.py` holds both prompt sets so the figure and the regression
suite measure the same thing. Sweep `--ngram-min-match` too: it is the
coverage/acceptance dial and the shape of that curve is the most interesting
thing the n-gram proposer has to say.

Calibrate expectations: a 0.6B target on a laptop GPU is close to the least
favourable case for this technique, whose reputation comes from 7B+ models where
fixed overheads vanish. A realistic landing zone here is 1.3-1.6x, not 3x.

Keep `experiments/spec_breakeven.py --end-to-end` as the cross-check on any
throughput claim — a microbenchmark that disagrees with `Engine.generate` is
measuring an artefact, and this one was caught doing exactly that once already.

Two things to respect, both measured:

- **The backend patch must be applied.** `patches/README.md`. Without it decode
  is nondeterministic above width 6 and every gate here becomes meaningless.
- **Byte-identical greedy output is a tolerance, not a guarantee.** It holds in
  every configuration measured so far (see Phase 5 Results), but batch shape
  moves logits by up to 0.5 absolute and a near-tie can flip an emitted token
  without anything being wrong. Keep the near-tie classification rather than
  asserting bitwise equality blindly. This is cuBLAS retiling the linear layers
  per batch shape; it is unrelated to the decode race and was not fixed by the
  patch.

## Phase Checklist

- Phase 0: **complete — go decision recorded.** Note its identified verify path
  was wrong; corrected in Phase 4. Committed as `8f16197` on branch `test`.
- Phase 1: **optional, and superseded as the route to a speedup.** Acceptance
  rate was the only thing between this feature and a speedup, and the **n-gram
  proposer cleared that bar in half a day instead of a week** — it needs ~30%
  acceptance rather than ~60%, and measures ~1.7x on repetitive text. Phase 1
  remains worth doing for the open-ended case that lookup cannot help (where
  n-gram is a ~10% *loss*); if so, distil from Qwen3-0.6B rather than
  pretraining on a general corpus. RESUME HERE step 2.
- **N-gram proposer: complete, measured and gated (not a numbered phase).**
  `speculative_method="ngram"`, `minivllm/executor/ngram.py`,
  `tests/test_ngram_proposer.py`. Break-even ~30%; ~1.7x on repetitive prompts,
  ~0.9x on open-ended. See the N-gram Proposer section.
- Phase 2: **complete — both gates passing**, 18/18 exact match. See results above.
- Phase 3: **complete and validated on GPU.** Config fields,
  `load_draft_model`, dual KV-cache allocation, draft checkpoint. Exit
  criterion passed byte-identically; see Phase 3 Validation.
- Phase 4: **complete and gated on GPU.** `Executor.verify()` routed through
  `flash_attn_with_kvcache` with K+1 tokens in the batch dimension — *not*
  through `flash_attn_varlen_func`, which cannot express the shape. See Phase 4
  Results.
- Phase 5: **complete and gated on GPU.** `Sampler.rejection_sample`,
  `Executor.propose` / `execute_speculative`, multi-token commit through the
  scheduler, acceptance-rate metrics. Greedy output byte-identical with
  speculation on and off at K=4 and K=8, at 0% and 100% acceptance. See Phase 5
  Results.
- Phase 6: **complete.** `tests/test_spec_decode_correctness.py` and
  `tests/test_spec_decode_end_to_end.py`, three tiers, mutation-checked, plus a
  root `conftest.py` and `tests/spec_harness.py`. The pre-existing
  `test_block_manager.py` failure is fixed, so the suite is green. Its exit
  criterion survives as written but as a **tolerance**: byte-identical greedy
  output holds everywhere measured, though a near-tie flip is possible and is
  not a bug. See Phase 6 Results.
- Phase 7: **complete.** `benchmark/spec_sweep.py` sweeps four proposers against
  K on two workloads, one process per configuration, and renders
  `docs/spec_sweep.svg`; the README carries the write-up. Headline: **1.81x on
  repetitive text at K=4**, 0.94x on open-ended. The sweep's output-hash check
  found the KV batch-width ceiling — see Phase 7 Results.

## Handoff Notes for Agents

- **Start at "RESUME HERE" above.** The environment is built, speculative
  decoding runs end to end through Phase 6, and the n-gram proposer gives it a
  measured speedup on repetitive workloads. Next work is Phase 7 (step 3).
- Read [speculative-decoding-plan.md](speculative-decoding-plan.md) first.
- [experiments/engine_spec_gate.py](experiments/engine_spec_gate.py) is the
  engine-level correctness gate and the fastest way to confirm the stack is
  healthy. It carries the 4 GB config profile; the stock `Config` will not
  start on this GPU.
- [docs/spec_decoding_feasibility.md](docs/spec_decoding_feasibility.md) is the
  authoritative note; it records the go decision and the two remaining
  empirical checks on paged-varlen semantics.
- The greedy exact-match harness is the correctness gate at every level —
  prototype (Phase 2) and engine (Phase 6), both done. Any new proposer must
  keep `pytest tests/ --slow` green; a proposer changes acceptance rate, never
  output.
- Do not modify paging or scheduler core code unless the plan is explicitly revised.
