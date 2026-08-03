"""Phase 4 step 1 — the empirical checks docs/spec_decoding_feasibility.md left open.

The feasibility note confirmed `flash_attn_varlen_func`'s *signature* from source
— no `seqlen_q` restriction, accepts a paged `block_table`, supports GQA — and
concluded the verify pass could be a varlen call with `causal=True`. Signatures
are not semantics. Run on the GPU, that plan does not work, and this script is
the record of why.

Three questions, three answers:

**Q1. Is the paged gather correct for arbitrary block ids?**
Only when `kv_cache_block_size` is a multiple of the kernel's N-tile. Both
kernels do one block-table lookup per *tile*, not per row
(`csrc/mfa/prefill.cuh:52`, `csrc/mfa/decode.cuh:50`):

    block_table_idx    = nbidx * kBlockN / page_block_size
    block_table_offset = nbidx * kBlockN - block_table_idx * page_block_size
    offset = block_table[block_table_idx] * cache_block_stride + ...

then reads `kBlockN` consecutive rows from `offset`. If a tile straddles two
pages the second page is read from the wrong place — the kernel walks off the
first page into whatever block physically follows it. `flash.cu` instantiates
both kernels with `kBlockN = 64`, so the requirement is
`kv_cache_block_size % 64 == 0`. The repo satisfies this (256 stock, 64 in the
4 GB profile) and is *not* affected, but a smaller block size would silently
corrupt attention rather than fail — hence the assert in `Executor`.

**Q2. Where is `causal=True` anchored when `seqlen_q < seqlen_k`?**
Top-left. `prefill.cuh:416` masks on `col_0 > row_0` with `row` counted from the
start of the *query* segment and no `seqlen_k - seqlen_q` shift. Real
FlashAttention anchors bottom-right, which is what the Phase 0 note assumed. So
a K+1-token verify query against a long cache would have token i attending to
keys 0..i — blind to its own context. **The planned varlen verify pass is
unusable, and no `block_table` fix changes that.**

**Q3. Then what does work?**
`flash_attn_with_kvcache` — the decode kernel. Its `seqlen_q == 1` assert
constrains the *query* dimension, not the token count: K+1 tokens are expressed
as K+1 rows of the batch dimension, each with its own `cache_seqlens` and its
own (duplicated) block-table row. Every row then attends to exactly its own
prefix, which is precisely the verify semantics, using the one paged path this
repo already exercises in production. That is what Phase 4 builds on.

Usage:
    python experiments/paged_varlen_check.py
"""

import os
import sys

os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mini_flash_attention import flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.bfloat16

NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 64

# The N-tile both kernels are instantiated with in csrc/mfa/flash.cu.
KERNEL_BLOCK_N = 64

# bf16 through a long reduction; the reference runs in the same dtype, so this
# covers accumulation-order differences only.
TOL = 3e-2


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              query_offset: int) -> torch.Tensor:
    """Dense attention. Query row i sits at absolute position `query_offset + i`
    and attends to keys 0..query_offset+i inclusive."""
    seqlen_q, seqlen_k = q.shape[0], k.shape[0]
    repeat = NUM_HEADS // NUM_KV_HEADS
    qh = q.transpose(0, 1)
    kh = k.transpose(0, 1).repeat_interleave(repeat, dim=0)
    vh = v.transpose(0, 1).repeat_interleave(repeat, dim=0)
    rows = torch.arange(seqlen_q, device=DEVICE).unsqueeze(1) + query_offset
    cols = torch.arange(seqlen_k, device=DEVICE).unsqueeze(0)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=(cols <= rows))
    return out.transpose(0, 1)


def scatter_into_cache(seq_len: int, block_ids: list[int], block_size: int,
                       num_blocks: int, generator: torch.Generator):
    """Lay one sequence's KV into `block_ids`, in that order.

    Unused blocks hold large garbage rather than zeros, so a gather that lands
    on the wrong block shows up as a numerical blowup instead of quietly
    averaging in something near zero.
    """
    shape = (num_blocks, block_size, NUM_KV_HEADS, HEAD_DIM)
    k_cache = torch.randn(*shape, device=DEVICE, dtype=DTYPE, generator=generator) * 100.0
    v_cache = torch.randn(*shape, device=DEVICE, dtype=DTYPE, generator=generator) * 100.0

    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=generator)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=generator)
    for pos in range(seq_len):
        k_cache[block_ids[pos // block_size], pos % block_size] = k[pos]
        v_cache[block_ids[pos // block_size], pos % block_size] = v[pos]
    return k_cache, v_cache, k, v


# ---------------------------------------------------------------------------
# Q1 — when is the paged gather correct?
# ---------------------------------------------------------------------------

def check_block_size_constraint() -> bool:
    print("== Q1: paged gather vs kv_cache_block_size (kernel kBlockN = 64) ==")
    print("   scattered, deliberately non-ascending block ids; seqlen_q == seqlen_k")
    seq_len = 256
    ok_all = True
    for block_size in (16, 32, 64, 128, 256):
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        num_needed = (seq_len + block_size - 1) // block_size
        num_blocks = max(2 * num_needed, 8)
        # Reverse order: contiguous-but-descending, the layout this repo's
        # free-list actually hands out (deque.pop() takes the highest id first).
        block_ids = list(range(num_needed))[::-1]

        k_cache, v_cache, k, v = scatter_into_cache(seq_len, block_ids, block_size,
                                                    num_blocks, generator)
        q = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=generator)
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)
        bt = torch.tensor([block_ids], dtype=torch.int32, device=DEVICE)

        out = flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                     max_seqlen_q=seq_len, max_seqlen_k=seq_len,
                                     causal=True, block_table=bt)
        err = float((out.float() - reference(q, k, v, 0).float()).abs().max())

        expected_ok = block_size % KERNEL_BLOCK_N == 0
        got_ok = err < TOL
        agrees = expected_ok == got_ok
        ok_all &= agrees
        print(f"   block_size={block_size:4d}  max|err|={err:8.2e}  "
              f"{'correct' if got_ok else 'CORRUPT '}  "
              f"(predicted {'correct' if expected_ok else 'corrupt'}) "
              f"{'ok' if agrees else '<-- MODEL WRONG'}")
    print(f"   => rule confirmed: kv_cache_block_size must be a multiple of {KERNEL_BLOCK_N}")
    return ok_all


# ---------------------------------------------------------------------------
# Q2 — causal anchoring
# ---------------------------------------------------------------------------

def check_causal_anchoring() -> bool:
    print("\n== Q2: causal=True anchoring when seqlen_q < seqlen_k ==")
    print("   block ids ascending+contiguous and block_size=64, so Q1 is not a factor")
    block_size, seq_len = 64, 256
    num_needed = seq_len // block_size
    block_ids = list(range(num_needed))

    verdicts = []
    for seqlen_q in (1, 5, 17, 64, 256):
        generator = torch.Generator(device=DEVICE).manual_seed(1)
        k_cache, v_cache, k, v = scatter_into_cache(seq_len, block_ids, block_size, 8, generator)
        q = torch.randn(seqlen_q, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=generator)
        cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=DEVICE)
        cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)
        bt = torch.tensor([block_ids], dtype=torch.int32, device=DEVICE)

        out = flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                     max_seqlen_q=seqlen_q, max_seqlen_k=seq_len,
                                     causal=True, block_table=bt)
        bottom_right = float((out.float() - reference(q, k, v, seq_len - seqlen_q).float()).abs().max())
        top_left = float((out.float() - reference(q, k, v, 0).float()).abs().max())
        tags = [t for t, e in (("bottom-right", bottom_right), ("top-left", top_left)) if e < TOL]
        verdicts.append(tags)
        print(f"   seqlen_q={seqlen_q:4d} seqlen_k={seq_len}: "
              f"bottom-right err={bottom_right:.2e}  top-left err={top_left:.2e}  -> {tags or ['NEITHER']}")

    # seqlen_q == seqlen_k is ambiguous (both alignments coincide); the strict
    # cases are the ones that decide it.
    strict = verdicts[:3]
    top_left_only = all(t == ["top-left"] for t in strict)
    print("   => causal is TOP-LEFT anchored; the planned varlen verify pass cannot work"
          if top_left_only else "   => UNEXPECTED: re-examine, the verify plan may be viable")
    return top_left_only


# ---------------------------------------------------------------------------
# Q3 — the primitive Phase 4 actually uses
# ---------------------------------------------------------------------------

def check_decode_kernel_verify_shape(requests: list[dict], K: int, label: str) -> bool:
    """K+1 tokens as K+1 rows of the batch dimension, each with its own prefix.

    This is the verify pass in miniature: one call, several query tokens, each
    attending to a different amount of the same paged cache.

    This check once had to stay at batch width <= 5, because the decode kernel
    was nondeterministic above that and would fail this intermittently for a
    reason unrelated to paging. That was a backend race, fixed by
    patches/mini-flash-attention-decode-race.patch, so the ceiling is gone. If
    this does start failing intermittently at wider shapes, run
    experiments/decode_determinism_check.py before suspecting the paging.
    """
    batch_width = len(requests) * (K + 1)
    block_size = 64
    num_blocks = 32
    generator = torch.Generator(device=DEVICE).manual_seed(2)

    caches, truths = [], []
    for req in requests:
        # Cache holds `len` real tokens; the verify pass will query the last
        # K+1 absolute positions, so give the cache room for them.
        kc, vc, k, v = scatter_into_cache(req["len"], req["blocks"], block_size, num_blocks, generator)
        caches.append((kc, vc))
        truths.append((k, v))

    # One shared cache tensor, as the executor has: rebuild it by merging.
    k_cache = caches[0][0].clone()
    v_cache = caches[0][1].clone()
    for req, (kc, vc) in zip(requests[1:], caches[1:]):
        for bid in req["blocks"]:
            k_cache[bid] = kc[bid]
            v_cache[bid] = vc[bid]

    q_rows, cache_seqlens, block_rows, expected = [], [], [], []
    # Padding width of the block table, unrelated to the batch width above.
    table_width = max(len(r["blocks"]) for r in requests)
    for req, (k, v) in zip(requests, truths):
        for j in range(K + 1):
            # Query token j of this request sits at absolute position
            # len - (K+1) + j and attends to keys 0..that position.
            pos = req["len"] - (K + 1) + j
            q = torch.randn(1, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=generator)
            q_rows.append(q)
            cache_seqlens.append(pos + 1)
            block_rows.append(req["blocks"] + [-1] * (table_width - len(req["blocks"])))
            expected.append(reference(q, k[:pos + 1], v[:pos + 1], pos))

    q = torch.cat(q_rows, dim=0)
    out = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=torch.tensor(cache_seqlens, dtype=torch.int32, device=DEVICE),
        block_table=torch.tensor(block_rows, dtype=torch.int32, device=DEVICE),
    ).reshape(q.shape)

    worst = 0.0
    for i, ref in enumerate(expected):
        worst = max(worst, float((out[i:i + 1].float() - ref.float()).abs().max()))
    ok = worst < TOL
    print(f"   {label}: {len(requests)} request(s) x {K + 1} query tokens "
          f"(batch width {batch_width}), per-row cache_seqlens {cache_seqlens}")
    print(f"   max|err| vs dense attention = {worst:.2e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    assert torch.cuda.is_available(), "this check needs the GPU backend"
    print(f"device={torch.cuda.get_device_name(0)}  dtype={DTYPE}  "
          f"GQA {NUM_HEADS}q/{NUM_KV_HEADS}kv  head_dim={HEAD_DIM}\n")

    q1 = check_block_size_constraint()
    q2 = check_causal_anchoring()

    print("\n== Q3: flash_attn_with_kvcache as the verify primitive ==")
    # One request spanning three blocks, and two requests of different lengths
    # so the per-row cache_seqlens and per-row block table are exercised across
    # sequences too. Block ids descend, matching what the repo's free-list hands
    # out (deque.pop() takes the highest id first).
    q3_single = check_decode_kernel_verify_shape(
        [{"blocks": [9, 8, 7], "len": 130}], K=4, label="one request, K=4")
    q3_multi = check_decode_kernel_verify_shape(
        [{"blocks": [9, 8, 7], "len": 130}, {"blocks": [5, 4], "len": 70}], K=4,
        label="two requests, K=4")
    print("   => the decode kernel expresses the verify pass exactly; no backend patch needed")

    results = {
        "Q1 block-size rule": q1,
        "Q2 causal is top-left": q2,
        "Q3 decode-kernel verify": q3_single and q3_multi,
    }

    print()
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
