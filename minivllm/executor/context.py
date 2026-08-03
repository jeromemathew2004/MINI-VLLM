from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    Context holds information for the current forward pass in the executor.
    """
    
    
    prefill: bool = False



    """
    cumulative sequence lengths for queries and keys
    Used for flash attention implementation.
    
    for seqeunce lengths: [L1, L2, L3], the cumulative sequence lengths is [0, L1, L1+L2, L1+L2+L3]
    """
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None

    max_seqlen_q: int = 0
    max_seqlen_k: int = 0


    """
    positions of the tokens in the original sequence
    
    In continuous batching, all input sequences are concatenated into a single sequence.
    for example, we have 2 sequences:
    seq 0: [A B C D]
    seq 1: [E F G]
    
    after continuous batching, we have:
    [A B C D E F G]
    
    The positions tensor is:
    [0 1 2 3 0 1 2]
    
    this positions tensor keeps track of the original positions of the tokens in their original sequences.
    """
    positions: torch.Tensor | None = None



    """
    mapping from token index to slot index in the cache
    the index is the token index in the continuous batching sequence
    the value is the slot index in the kv cache
    """
    slot_mapping: torch.Tensor | None = None



    """
    The sequence lengths of the kv cache for each request.
    For example, if we have 2 requests:
    req 0 has kv cache sequence length of 4
    req 1 has kv cache sequence length of 3
    then the cache_seqlens is [4, 3]

    Note this indexes QUERY ROWS, which is the same thing as requests only
    because plain decode contributes exactly one query token per request. A
    speculative verify pass contributes K+1 query tokens per request, each
    attending to a different amount of that request's history, so it emits K+1
    entries per request -- one per token, holding that token's own prefix
    length. See Executor._build_verify_input.
    """
    cache_seqlens: torch.Tensor | None = None


    """
    The block table is a 2D tensor that maps each request to its corresponding block in the cache.
    
    For example, req 0 uses blocks 0, 1, 2, 3
                 req 1 uses blocks 4, 5, 6

    The block table is:
    
    [
        [0, 1, 2, 3 ],  # req 0
        [4, 5, 6, -1],  # req 1
    ]

    One row per query row, matching cache_seqlens above: a verify pass repeats
    a request's row once per query token it contributes.
    """
    block_table: torch.Tensor | None = None

