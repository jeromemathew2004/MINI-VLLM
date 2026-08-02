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

    # whether to propose tokens with a draft model and verify them with the
    # target model, instead of decoding one token per request per step
    use_speculative_decoding: bool = False

    # the path of the draft model. must share the target's vocabulary
    draft_model: str = ""

    # the number of tokens the draft model proposes per step, "K" in the
    # literature. each step then emits between 1 and K+1 tokens per request
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

        if self.use_speculative_decoding:
            assert self.draft_model, "use_speculative_decoding requires draft_model to be set."
            self.draft_model = os.path.expanduser(self.draft_model)
            assert os.path.isdir(self.draft_model), f"Draft model path {self.draft_model} is not a directory."
            assert self.num_speculative_tokens >= 1, "num_speculative_tokens must be at least 1."

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
