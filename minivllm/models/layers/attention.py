import torch
from torch import nn
import triton
import triton.language as tl

# from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from mini_flash_attention import flash_attn_with_kvcache, flash_attn_varlen_func
from minivllm.executor.context import Context


# this file is copied from nano-vllm with minor modifications

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class FlashAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scaling,
        num_kv_heads,
        sliding_window = (-1, -1),
    ):
        """
        
        window_size: (left, right). If not (-1, -1), implements sliding window local attention. 
                     left and right indicate how many tokens to the left/right each query can attend to.
        """
        
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, ctx: Context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        
        k_cache, v_cache = self.k_cache, self.v_cache
        
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)
            
        if ctx.prefill:
            # if ctx.block_table is not None, this means we are using prefix caching,
            # in this situation, we need to provide the full k_cache and v_cache and
            # the block_table to flash attention
            if ctx.block_table is not None:
                # The block table was documented as required here but never
                # actually passed, so the kernel read the paged cache as if it
                # were contiguous. Passing it is the fix for that bug -- but it
                # is not enough to make prefix caching correct, hence the
                # assert below.
                #
                # This kernel anchors its causal mask TOP-LEFT: it masks on
                # `col > row` with row counted from the start of the query
                # segment and no seqlen_k - seqlen_q shift (csrc/mfa/prefill.cuh
                # in w4096/mini-flash-attention, measured in
                # experiments/paged_varlen_check.py). A prefix-cached prefill
                # feeds only the uncached suffix as queries against the full
                # key sequence, which needs BOTTOM-RIGHT anchoring; under
                # top-left every query attends to the wrong keys. Fail loudly
                # rather than return quietly wrong attention.
                assert ctx.max_seqlen_q == ctx.max_seqlen_k, (
                    "prefix-caching prefill (seqlen_q < seqlen_k) needs a bottom-right "
                    "anchored causal mask, which mini-flash-attention does not implement. "
                    "Run KVCacheBlockManager with support_prefix_cache=False, or see "
                    "experiments/paged_varlen_check.py for the measurement."
                )
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=ctx.max_seqlen_q, cu_seqlens_q=ctx.cu_seqlens_q,
                                       max_seqlen_k=ctx.max_seqlen_k, cu_seqlens_k=ctx.cu_seqlens_k,
                                       causal=True,
                                       block_table=ctx.block_table)
        else:
            # Decode, and also the speculative verify pass. The kernel asserts
            # seqlen_q == 1, which constrains the *query* dimension and not the
            # number of tokens in flight: a verify pass passes its K+1 tokens
            # per request as K+1 separate rows of the batch dimension, each
            # carrying its own cache_seqlens and its own copy of the request's
            # block-table row. See Executor._build_verify_input.
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=ctx.cache_seqlens, block_table=ctx.block_table)
        return o
