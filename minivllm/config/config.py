import os
from typing import Set
from dataclasses import dataclass, field
from transformers import AutoConfig, PretrainedConfig

@dataclass()
class Config:
    # ================= model config =================

    # the model name or path
    model: str = "~/huggingface/Qwen3-0.6B"

    # the huggingface config of the model
    hf_config: PretrainedConfig = field(init=False)

    # the max length of generated tokens
    max_model_len: int = 4096




    # ================= scheduler config =================

    # the max number of tokens in a batch when running prefill and decode
    max_num_batched_tokens: int = 16384

    # the max number of sequences in a batch when running prefill and decode
    max_num_batched_seqs: int = 512

    # the end of sequence token id
    eos_token_ids: Set[int] = None


    # ================= kv cache config =================
    # the number of kv cache blocks
    kv_cache_num_blocks: int = 256
    # the size of each kv cache block
    kv_cache_block_size: int = 256
    # the max utilization of the kv cache memory
    gpu_memory_utilization: float = 0.5
    
    
    
    # ================= executor config =================
    # whether to use cuda graph for decoding
    use_cuda_graph:  bool = True



    # ================= speculative decoding config =================

    # whether to propose tokens ahead of time and verify them with the target
    # model, instead of decoding one token per request per step
    use_speculative_decoding: bool = False

    # where the proposals come from. "draft" runs a small model K times;
    # "ngram" looks the last few tokens up in the request's own history and
    # proposes what followed them before, which needs no model, no KV cache and
    # no GPU time. both feed the same verify-and-accept core, because a greedy
    # draft and a lookup both produce a one-hot proposal distribution
    speculative_method: str = "draft"

    # the path of the draft model. must share the target's vocabulary.
    # required by speculative_method="draft", unused by "ngram"
    draft_model: str = ""

    # the longest token suffix speculative_method="ngram" tries to match
    # against the history. it works down from here, so a larger value only adds
    # stronger matches, never removes weaker ones
    ngram_max_match_len: int = 3

    # the shortest match worth proposing from. this is a quality floor, not a
    # performance knob: a 1-token match exists nearly everywhere, so a low floor
    # makes the proposer speculate on almost every step regardless of evidence,
    # and a round costs ~1.4x a plain decode step. measured on the dev host
    # (experiments/spec_breakeven.py --end-to-end ngram_graph), 3 beat 2 on both
    # workloads — it roughly halves how often the proposer fires while raising
    # acceptance, and the rounds it drops were the losing ones. both sides of
    # that trade are reported by Metrics as speculation_rate and acceptance_rate
    ngram_min_match_len: int = 3

    # the number of tokens proposed per step, "K" in the literature. each step
    # then emits between 1 and K+1 tokens per request
    num_speculative_tokens: int = 4

    # the huggingface config of the draft model
    draft_hf_config: PretrainedConfig = field(init=False, default=None)


    def __post_init__(self):
        # the defaults above are written with a leading "~", which only a unix
        # shell expands — do it here so the defaults work as given
        self.model = os.path.expanduser(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.eos_token_ids is None:
            assert hasattr(self.hf_config, "eos_token_id")
            eos_token_id = self.hf_config.eos_token_id
            if isinstance(eos_token_id, int):
                self.eos_token_ids = {eos_token_id}
            else:
                self.eos_token_ids = set(eos_token_id)
                
        assert os.path.isdir(self.model), f"Model path {self.model} is not a directory."

        # mini-flash-attention resolves ONE block-table entry per 64-key tile
        # rather than per key row (csrc/mfa/{prefill,decode}.cuh; both kernels
        # are instantiated with kBlockN=64 in csrc/mfa/flash.cu), then reads 64
        # consecutive rows from there. A tile that straddles two pages
        # therefore reads the second half from whatever block physically
        # follows the first -- silently wrong attention, not a crash. Measured
        # in experiments/paged_varlen_check.py.
        assert self.kv_cache_block_size % 64 == 0, (
            f"kv_cache_block_size must be a multiple of 64, got {self.kv_cache_block_size}. "
            f"The attention kernel resolves one block-table entry per 64-key tile, so a "
            f"smaller page silently corrupts attention for any sequence past its first page."
        )

        if self.use_speculative_decoding:
            assert self.speculative_method in ("draft", "ngram"), (
                f"unknown speculative_method {self.speculative_method!r}; "
                f"expected 'draft' or 'ngram'."
            )
            assert self.num_speculative_tokens >= 1, "num_speculative_tokens must be at least 1."

        if self.use_speculative_decoding and self.speculative_method == "ngram":
            assert 1 <= self.ngram_min_match_len <= self.ngram_max_match_len, (
                f"need 1 <= ngram_min_match_len ({self.ngram_min_match_len}) <= "
                f"ngram_max_match_len ({self.ngram_max_match_len})."
            )
            # Nothing else to set up: the proposer is a search over
            # `req.tokens`, so there is no second model to load, no second KV
            # cache to charge against the memory budget and no draft config.

        if self.use_speculative_decoding and self.speculative_method == "draft":
            assert self.draft_model, "speculative_method='draft' requires draft_model to be set."
            self.draft_model = os.path.expanduser(self.draft_model)
            assert os.path.isdir(self.draft_model), f"Draft model path {self.draft_model} is not a directory."

            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model)

            # Rejection sampling compares the draft's probability and the
            # target's probability for the *same* token id, so the two models
            # must index the same vocabulary. A mismatch here is silent
            # corruption rather than a crash, hence the assert.
            assert self.draft_hf_config.vocab_size == self.hf_config.vocab_size, (
                f"Draft vocab size {self.draft_hf_config.vocab_size} does not match "
                f"target vocab size {self.hf_config.vocab_size}."
            )

            # Both models are built under a single torch default dtype, which
            # the executor takes from the target. Force the draft's config to
            # agree so its KV cache is sized in the dtype it is allocated in.
            self.draft_hf_config.dtype = self.hf_config.dtype
