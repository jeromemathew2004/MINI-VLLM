# Speculative Decoding for mini-vllm — Implementation Spec

**Audience:** this document is written to be handed directly to a coding LLM
(Claude Code, or any other agent with repo access) as a work order. It
assumes the agent has the `mini-vllm` repository open and can read/edit
files. It references real file paths and real class/function names from the
current codebase — verify each reference against the live code before
editing, since the repo will have moved on since this doc was written.

**Do not start writing integration code until Phase 0 and Phase 1 are both
complete and their exit criteria are met.** This project has failed before
for other people by skipping straight to "wire it into the engine" and
discovering a wrong assumption three layers deep.

---

## 1. What we're building and why

mini-vllm currently decodes exactly one token per request per step (see
`Executor._build_decode_input` in `minivllm/executor/executor.py` — it
appends `req.tokens[-1]`, a single token, and builds a `Context` sized for
one new position per request).

We're adding **speculative decoding**: a small, fast "draft" model proposes
`K` tokens ahead for each request; the real target model (Qwen3-0.6B)
verifies all `K` proposed tokens in a single forward pass; a rejection
sampling step decides how many of the `K` tokens to actually accept. When
acceptance is good, this turns `K` sequential expensive forward passes into
1 expensive pass + 1 cheap pass, which is where the speedup comes from.

**Critical property, not optional:** correctly implemented rejection
sampling (Leviathan et al., 2023 / Chen et al., 2023 — "Fast Inference from
Transformers via Speculative Decoding" / "Accelerating Large Language Model
Decoding with Speculative Sampling") produces output that is **exactly**
distributed as if you had sampled from the target model alone, token by
token, with no draft model involved. This is not an approximation. At
temperature 0 (greedy), this means: **speculative decoding output must be
byte-for-byte identical to non-speculative greedy decoding output, for the
same prompt.** This is your correctness harness — not a nice-to-have, the
central design constraint. If your implementation doesn't satisfy this, the
rejection sampling math is wrong, full stop, regardless of how fast it runs.

---

## 2. Repo context the agent needs

Relevant current files (read all of these before writing anything):

- `minivllm/executor/executor.py` — `Executor` class. Owns the model,
  builds prefill/decode inputs, calls forward, samples. This is where most
  new code goes.
- `minivllm/executor/context.py` — `Context` dataclass carrying per-forward
  metadata (`positions`, `cu_seqlens_q/k`, `slot_mapping`, `cache_seqlens`,
  `block_table`) consumed by `FlashAttention.forward` in
  `minivllm/models/layers/attention.py`.
- `minivllm/models/layers/attention.py` — wraps `flash_attn_varlen_func`
  (prefill) and `flash_attn_with_kvcache` (decode) from the
  `mini_flash_attention` package.
- `minivllm/scheduler/scheduler.py`, `minivllm/scheduler/batch.py` — FCFS
  scheduling, `Batch(PREFILL|DECODE, requests)`.
- `minivllm/kvcache/block_manager.py` — `KVCacheBlockManager`, paged
  allocation with prefix-cache hashing.
- `minivllm/engine/request.py` — `Request`, tracks `tokens`, `blocks`,
  `num_cached_tokens`, `state`.
- `minivllm/models/layers/sampler.py` — `Sampler`, currently returns a
  single sampled/argmax token per row of logits. **You will need a second
  sampling path here for rejection sampling** — do not overwrite the
  existing greedy/top-k/top-p path used by non-speculative decoding.
- `minivllm/config/config.py` — add new fields here (see Phase 3).

**Do not modify:** `block_manager.py`'s hashing/refcounting logic or
`scheduler.py`'s prefill scheduling. Speculative decoding is a decode-path
feature — scope creep into the prefill/paging core is out of bounds for
this task.

---

## 3. Phase 0 — Validate assumptions before building anything

This phase produces no product code. It produces a go/no-go decision and a
short written note (`docs/spec_decoding_feasibility.md`) recording what you
found. Do not skip it — every subsequent phase depends on these answers.

1. **Does `flash_attn_with_kvcache` in the `mini_flash_attention` package
   support more than one new query token per sequence in a single call?**
   Real FlashAttention's `flash_attn_with_kvcache` supports a `q` tensor
   with `seqlen > 1` for exactly this use case (that's literally what it's
   for in production speculative decoding). But `mini_flash_attention` is a
   pulled-from-GitHub minimal reimplementation — verify its signature and
   behavior directly (read its source, or write a 10-line standalone script
   that calls it with `q.shape = [batch, K+1, num_heads, head_dim]` against
   a small cache and checks the output shape and correctness against a
   manual/reference attention computation). If it only supports
   `seqlen_q == 1`, that changes Phase 4 significantly (you'd need to patch
   `mini_flash_attention` itself, which is a bigger task — flag this back
   before proceeding).
2. **VRAM check.** Load Qwen3-0.6B and a placeholder tiny model
   (any small HF causal LM, doesn't need to be your final draft model yet)
   simultaneously on your GPU. Confirm both fit with room for KV-cache and
   activations at your target batch size. Record actual numbers.
3. **Record the exit criteria in `docs/spec_decoding_feasibility.md`:**
   confirmed multi-token decode support (or a documented workaround plan),
   and confirmed VRAM headroom.

---

## 4. Phase 1 — Train the draft model (standalone, reuses Project #2)

Build this **completely outside mini-vllm**, as its own trainable script —
do not import anything from `minivllm/` yet.

1. **Tokenizer:** use Qwen3-0.6B's actual tokenizer (`AutoTokenizer.from_pretrained`
   on the same model path/repo). The draft model's vocabulary must match
   the target model's exactly — this is not optional, the rejection
   sampling math operates over the same token distribution for both models.
2. **Architecture:** small dense decoder-only transformer, RoPE + RMSNorm +
   SwiGLU (mirror Qwen3's block shape so the training/inference code is
   familiar), but far fewer layers/smaller hidden dim. Target **10–30M
   non-embedding parameters** — note the embedding + LM head at Qwen3's
   ~150k vocab will dominate parameter count at this scale, budget for that
   (consider tying input embedding and output head weights to control this).
3. **Data:** doesn't need to match Qwen3's training data — a general web
   text or instruction-style corpus is fine. The draft model's job is
   "propose plausible continuations," not "be a good model" — its
   usefulness is measured entirely by acceptance rate against the target,
   not by its own perplexity.
4. **Training loop:** standard next-token cross-entropy, same as Project #2.
   Save checkpoints in a HF-compatible format (`save_pretrained`) so it
   loads the same way Qwen3 does elsewhere in this repo.
5. **Exit criteria:** a saved draft model checkpoint + tokenizer, plus a
   short standalone script that loads it and generates text, confirming
   sane (not necessarily excellent) completions.

---

## 5. Phase 2 — Standalone draft+verify prototype (still outside the engine)

Before touching `minivllm/`, prove the algorithm is correct in isolation.
Write a single script, e.g. `experiments/spec_decode_prototype.py`, using
plain HuggingFace `generate`-adjacent calls (not the mini-vllm engine, not
paging) — just two models and manual loops.

1. Implement the core loop for one sequence at a time:
   - Draft model autoregressively proposes `K` tokens (own tiny KV-cache,
     doesn't need paging at this stage).
   - Target model runs one forward pass over the `K` proposed tokens (plus
     the current context) and returns logits/probabilities at each of the
     `K` positions.
   - Implement the **modified rejection sampling algorithm** exactly as
     specified in Leviathan et al. / Chen et al.: for each position `i`
     from 1 to `K`, compare target probability `p(x_i)` vs draft
     probability `q(x_i)` for the token the draft proposed; accept with
     probability `min(1, p(x_i)/q(x_i))`; on first rejection, resample from
     the residual distribution `max(0, p - q)` (renormalized) and stop; if
     all `K` accepted, sample one bonus token from the target's `K+1`-th
     position distribution.
2. **Correctness test, greedy case:** with temperature 0 (both models
   argmax/greedy — note greedy rejection sampling degenerates to: accept
   token `i` iff draft's argmax equals target's argmax at that position,
   reject and take target's argmax otherwise), generate N tokens for a
   fixed prompt with (a) plain target-only greedy decoding and (b) your
   spec-decoding loop. **Assert token-for-token exact match.** This is a
   hard gate — do not proceed to Phase 3 until this passes for several
   different prompts and several different `K` values (try `K` = 2, 4, 8).
3. **Exit criteria:** exact-match test passing consistently, plus a printed
   acceptance rate (average fraction of draft tokens accepted per round) so
   you have a baseline number before engine integration adds any overhead
   or bugs.

---

## 6. Phase 3 — Wire the draft model into the engine's model loading

1. Add to `Config` (`minivllm/config/config.py`): `use_speculative_decoding: bool`,
   `draft_model: str` (path), `num_speculative_tokens: int` (this is `K`).
2. Decide and document the draft model's KV-cache strategy. Simplest
   correct option for a first version: give the draft model its **own
   separate, non-paged, contiguous KV-cache** (a plain tensor sized for
   `max_num_batched_seqs × max_model_len`, no block manager involvement).
   It's small enough that paging it isn't worth the complexity yet — treat
   full paging of the draft cache as a documented stretch goal, not part of
   this phase.
3. Load both models in `Executor.__init__` when `config.use_speculative_decoding`
   is set. Keep the target model's existing paged-cache path completely
   unchanged.

---

## 7. Phase 4 — Multi-token verify pass in the target model

This is the phase most likely to surface the Phase 0 assumption problem —
if it does, stop and resolve that before continuing.

1. Extend `Context` (or add a parallel path) to describe a decode step
   where each request contributes `K+1` new query positions instead of 1 —
   `positions`, `slot_mapping`, and `cache_seqlens` all need to account for
   this. Read `Executor._build_decode_input` closely; you're building a
   variant of it, not modifying prefill.
2. Confirm the block manager allocates enough blocks for `K+1` new tokens
   per request per step (`allocate_block_for_decode` currently allocates
   for 1 new token's worth of space — check `request_required_blocks`
   handles a request that just grew by `K+1` tokens, not 1).
3. **The hard correctness issue to solve explicitly:** the verify forward
   pass writes KV-cache entries for all `K+1` proposed positions (via
   `store_kvcache` inside `FlashAttention.forward`), but if rejection
   sampling only accepts `M < K` of them, the cache now contains stale
   KV entries for tokens that were never actually accepted into the
   sequence. Document and implement your chosen fix — options include:
   truncating `req.tokens` and treating the unaccepted cache slots as
   logically overwritten on the next round (relies on `cache_seqlens`
   being the source of truth for how much of the cache is "real," not the
   physical slot contents), or explicitly zeroing/invalidating the
   rejected slots. Pick one, document why in a code comment, and test it
   under partial-acceptance and zero-acceptance (first token rejected)
   cases specifically — zero-acceptance is the edge case most
   implementations get wrong first.
4. CUDA graphs: `CudaGraphRunner` currently captures graphs for fixed
   *batch sizes* at decode time assuming 1 new token per request. A
   multi-token decode step changes the input shape. Simplest first version:
   **disable CUDA graph replay for speculative decode steps** (fall back to
   eager execution) and note this as a known limitation/follow-up rather
   than trying to solve graph capture for variable-length verify passes in
   this phase.

---

## 8. Phase 5 — Rejection sampling in the engine + bookkeeping

1. Add the rejection sampling logic from Phase 2 as a new method on
   `Sampler` (`minivllm/models/layers/sampler.py`) — something like
   `Sampler.verify(draft_tokens, draft_probs, target_logits) -> (accepted_tokens, num_accepted)`.
   Keep the existing `forward` (used by plain decode) untouched.
2. Update the decode step in `Executor`/`Engine.step` to: run draft
   proposal → run target verify pass → run rejection sampling → append
   however many tokens were actually accepted (`Request.append_output_token`,
   called `num_accepted` times, not always once) → update
   `req.num_cached_tokens` / block bookkeeping to match the *actual* accepted
   length, not `K+1`.
3. `Metrics` (`minivllm/engine/metrics.py`) currently assumes 1 decode token
   per request per step for throughput math. Extend it to count actual
   accepted tokens per step, and **add acceptance rate as a new tracked
   stat** — you'll want this for the benchmark chart in Phase 7.

---

## 9. Phase 6 — Correctness harness in the engine (not just the prototype)

Re-run the exact-match test from Phase 2, but now through the real engine
(`Engine.generate`, real scheduler, real paged cache), comparing
speculative vs non-speculative greedy output for the same prompts. This is
a stricter test than Phase 2's because it exercises the block manager,
scheduler, and CUDA-graph-disabled decode path together. Add this as a
persistent test file (e.g. `tests/test_spec_decode_correctness.py`) that
can be re-run after any future change — this is your regression guard, the
same role `tests/test_block_manager.py` already plays for paging.

---

## 10. Phase 7 — Benchmark

Extend `benchmark/run_mini_vllm.py` (or add a sibling script) to report,
for a sweep of `K` values (try 0/baseline, 2, 4, 8):

- Tokens/sec (as already measured elsewhere in the repo)
- **Acceptance rate** — the number that explains *why* the speedup does or
  doesn't materialize; a low acceptance rate with a high `K` will show
  wasted draft compute, and the chart should make that visible
- A plot of throughput vs. `K`, since there's a real tradeoff: too small a
  `K` wastes the opportunity, too large a `K` wastes compute on proposals
  that get rejected past the first divergence

This chart, alongside the correctness-harness pass, is the centerpiece of
what goes in the README and resume writeup for this feature — it should
show a clear "and here's exactly how much and why" result, not just a
single before/after number.

---

## 11. Definition of done

- [ ] Phase 0 feasibility note committed, go decision recorded
- [ ] Draft model trained and checkpointed, tokenizer confirmed matching Qwen3
- [ ] Standalone prototype passes exact-match greedy test across multiple prompts and `K` values
- [ ] Draft model loads alongside target model in `Executor` with documented (non-paged) cache strategy
- [ ] Multi-token verify pass implemented, partial/zero-acceptance cache handling explicitly tested
- [ ] Rejection sampling implemented as a separate `Sampler` path, existing greedy/top-k/top-p path unmodified
- [ ] Engine-level correctness test passes and is committed as a persistent regression test
- [ ] Benchmark sweep across `K` values with acceptance rate reported, chart generated
- [ ] README updated: new "Speculative Decoding" section explaining the feature, its results, and — per the attribution fix already in progress — clearly marked as original work (this feature has no nano-vllm equivalent to disclose against)
