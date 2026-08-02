# Speculative Decoding Feasibility Note

Original date: 2026-07-19
Revised: 2026-08-03 — **decision reversed from no-go to go**

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

## Still to validate empirically

The signature and docstring are confirmed; the following need a live GPU
check once the environment is built, and are the first things to test in
Phase 4:

1. Exact semantics of `cu_seqlens_k` when `block_table` is passed — whether
   it carries full per-sequence KV length (expected) and how the kernel maps
   logical position to `block_table[b][pos // block_size]`.
2. That the 4-D cache tensor `(num_blocks, block_size, num_kv_heads,
   head_dim)` is accepted directly in the `k`/`v` slots under the paged path.

Neither affects the go decision — they affect how the Phase 4 context is
built, and both are cheap to check with a small standalone script.
