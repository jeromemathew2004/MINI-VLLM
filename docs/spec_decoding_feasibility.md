# Speculative Decoding Feasibility Note

Original date: 2026-07-19
Revised: 2026-08-03 — **decision reversed from no-go to go**
Revised again: 2026-08-03 (Phase 4) — **go stands, but the mechanism below is
wrong**. The verify pass does *not* go through `flash_attn_varlen_func`. That
kernel's causal mask is top-left anchored, which the signature does not reveal
and which makes the shape unusable. The verify pass goes through the decode
kernel instead, with K+1 tokens in the batch dimension. See "Phase 4
measurements" at the bottom — that section supersedes the reasoning in the
middle of this document, which is kept because the record of *why* the wrong
conclusion looked right is worth having.

Status note, 2026-08-04: this is a **feasibility** document and stops being the
interesting one once the thing is built. Phase 5 has since landed and the engine
decodes speculatively end to end. `PROGRESS.md` is the source of truth for
status; nothing measured since contradicts the Phase 4 section below.

## Summary of the revision

The 2026-07-19 note closed Phase 0 as a **no-go**, on the grounds that
`mini_flash_attention.flash_attn_with_kvcache` asserts `seqlen_q == 1` and
therefore cannot run a multi-token verify pass.

That observation about the *decode* kernel is correct and still stands. The
conclusion drawn from it was wrong, because it assumed the verify pass must
go through the decode kernel. It does not.

**A verify pass is structurally a chunked-prefill pass**, not a decode pass:
each request contributes `K+1` query tokens attending to its own full KV
cache. That is exactly the shape `flash_attn_varlen_func` already handles,
and this repo already has a code path that calls it against the paged cache
(the prefix-caching branch in `FlashAttention.forward`).

So the multi-token verify pass needs **no kernel patch**. Phase 0 is a go.

## Confirmed from the backend source

Verified against `w4096/mini-flash-attention` @ `main`,
`mini_flash_attention/interface.py`:

```python
def flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=None, block_table=None, num_splits=0,
) -> torch.Tensor:
    assert q.size(1) == 1, "flash_attn_with_kvcache currently only supports seqlen_q=1 for decoding"
```

```python
def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    causal: bool = False,
    block_table=None,
) -> torch.Tensor:
```

Relevant properties of `flash_attn_varlen_func`:

- **No `seqlen_q` restriction.** No asserts in the function body at all.
- **Accepts a paged `block_table`**, documented as
  `(batch_size, max_blocks_per_seq), dtype torch.int32` — "uses the block
  table to find the correct blocks in k, v ... where the valid blocks in k, v
  may not be contiguous."
- **Supports GQA** — "passing in K, V with fewer heads than Q", which Qwen3
  requires (`num_attention_heads` > `num_key_value_heads`).
- Shapes are `q: (total_q, nheads, headdim)`, output `(total_q, nheads, headdim)`.

## Latent bug found in the existing prefix-cache path

`minivllm/models/layers/attention.py` currently reads:

```python
if ctx.block_table is not None:
    k, v = k_cache, v_cache
o = flash_attn_varlen_func(q, k, v,
                           max_seqlen_q=ctx.max_seqlen_q, cu_seqlens_q=ctx.cu_seqlens_q,
                           max_seqlen_k=ctx.max_seqlen_k, cu_seqlens_k=ctx.cu_seqlens_k,
                           causal=True)
```

The comment directly above it says the block table must be passed to flash
attention, but **`block_table=` is never actually passed**. When prefix
caching is on, `k`/`v` are swapped for the full paged cache while the kernel
is left to read that cache as if it were contiguous — which produces wrong
attention output.

This is currently masked because `KVCacheBlockManager.__init__` defaults to
`support_prefix_cache=False`, so the branch is dead in normal runs.

It matters for this project because the verify pass wants exactly this call
*with* the block table. Fixing it is a prerequisite for Phase 4, and it is a
real pre-existing correctness fix in its own right.

## GPU architecture check (added 2026-08-03)

`mini-flash-attention` requires **SM 80+** — "NVIDIA Ampere GPU or newer",
CUDA 11.8+, tested on CUDA 12.8. The host GPU reports **compute capability
8.6**, so it clears the floor.

Worth recording because it cuts off the obvious fallback: **free Colab and
free Kaggle both provide a T4 (sm_75)**, and Kaggle's alternative is a P100
(sm_60). Both are *below* SM 80. A cloud escape hatch would have to be a paid
Ampere-or-newer tier, so the local 4 GB laptop GPU is the only free
environment that can run this backend at all.

The risk is therefore the toolchain, not the hardware: the build compiles
CUTLASS with `-std=c++20` and the project's README only documents a Linux
build path, so the MSVC route is unexercised.

## VRAM check

- Host GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB total.
- Live `nvidia-smi` at revision time: 3055 MiB free.
- Qwen3-0.6B weights in bf16: ~1.2 GiB.
- Default `gpu_memory_utilization=0.5` budgets ~2 GiB, leaving a thin margin
  for KV cache plus a draft model. Workable, but `max_model_len` and
  `kv_cache_num_blocks` will need lowering for local runs. This is a tuning
  constraint, not a blocker.

## Phase 0 exit criteria

- **Multi-token decode support: confirmed**, via `flash_attn_varlen_func`
  with `block_table`, rather than via the decode kernel. No patch to
  `mini-flash-attention` required.
- **VRAM headroom: confirmed but tight**; see above.
- **Phase 0 outcome: go.**

## Phase 4 measurements (2026-08-03) — supersedes the above

Run `python experiments/paged_varlen_check.py`. The two questions this note
left open were answered, and answering them overturned the mechanism proposed
above.

### 1. `causal=True` is TOP-LEFT anchored — the varlen verify pass cannot work

This is the finding that matters. `csrc/mfa/prefill.cuh:416` masks on
`col_0 > row_0`, with `row` counted from the start of the query segment and
**no `seqlen_k - seqlen_q` shift**. Measured: with `seqlen_k=256`, a query
segment of 1, 5, 17 or 64 tokens matches a top-left reference to 7.8e-03 and
disagrees with a bottom-right reference by ~2.5. Only `seqlen_q == seqlen_k`
matches both, which is why the existing prefill path never exposed it.

Real FlashAttention anchors bottom-right, and the section above assumed
mini-flash-attention does too. It does not. Under top-left anchoring a K+1
token verify query attends to keys 0..j instead of to its own history — every
proposal would be scored blind to the prompt. No `block_table` fix changes
this; it is the mask, not the gather.

This also means the repo's prefix-caching prefill path is unfixable as written,
not merely missing its `block_table` argument. `attention.py` now passes the
block table *and* asserts `seqlen_q == seqlen_k`, so the path fails loudly
instead of returning quietly wrong attention.

### 2. The paged gather is correct only when the page is a multiple of 64

Both kernels resolve **one block-table entry per N-tile**, not per key row
(`prefill.cuh:52`, `decode.cuh:50`):

```
block_table_idx    = nbidx * kBlockN / page_block_size
block_table_offset = nbidx * kBlockN - block_table_idx * page_block_size
offset = block_table[block_table_idx] * cache_block_stride + ...
```

then read `kBlockN` consecutive rows from `offset`. `flash.cu` instantiates
both with `kBlockN = 64`. A tile straddling two pages reads its second half
from whatever block physically follows the first. Measured: block sizes 16 and
32 corrupt (max err 3.4e+02), 64/128/256 are exact. The repo is safe at 256
stock and 64 in the 4 GB profile, and `Config.__post_init__` now asserts the
rule so a smaller value fails instead of silently corrupting.

The 4-D cache tensor *is* accepted directly in the `k`/`v` slots, as hoped.

### 3. The verify pass goes through the decode kernel

`flash_attn_with_kvcache`'s `assert q.size(1) == 1` constrains the **query
dimension, not the token count**. K+1 tokens become K+1 rows of the *batch*
dimension, each carrying its own `cache_seqlens` and its own copy of the
request's block-table row, so each attends to exactly its own prefix — which is
the verify semantics, exactly. Measured against dense attention: `0.00e+00`.

One forward pass over `num_requests * (K+1)` tokens, one GEMM per projection;
only the attention gather is per-row. No backend patch needed, and the paged
path used is the one the repo already exercises in production.

Also worth recording: **without** a `block_table`, `flash_attn_varlen_func`
requires `total_k == total_q` ("k must have shape (total_q_len, kv_num_heads,
head_dim)"), so the non-paged escape hatch does not exist either.

### 4. Unrelated blocker found: decode is nondeterministic at batch width >= 6

Not a speculative-decoding bug, and it reproduces on a clean checkout. See
`experiments/decode_determinism_check.py` and the PROGRESS.md entry. It caps
the usable K, because a verify pass runs at width `num_requests * (K+1)`.
