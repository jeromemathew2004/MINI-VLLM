import torch
import logging

from minivllm.config.config import Config
from minivllm.engine.request import Request
from minivllm.executor.context import Context
from minivllm.models.loader import load_model, load_draft_model
from minivllm.scheduler.batch import Batch
from minivllm.models.layers.sampler import Sampler
from minivllm.executor.graph import CudaGraphRunner

class Executor:
    def __init__(self, config: Config):
        self.config = config

        torch.set_default_device("cuda")
        torch.set_default_dtype(config.hf_config.dtype)
        self.model = load_model(config)

        # Loaded before _warmup_model so that the free/peak memory numbers
        # _init_kv_cache reads already account for the draft's weights.
        self.draft_model = load_draft_model(config) if config.use_speculative_decoding else None
        self.draft_kv_cache = None

        self.sampler = Sampler()

        self._warmup_model()

        self._init_kv_cache()
        
        if config.use_cuda_graph:
            logging.info("Initializing CUDA graph executor...")
            self.graph_runner = CudaGraphRunner(self.model, self.config, self.config.max_num_batched_seqs)
            self.graph_runner.capture()

        
    @staticmethod
    def _kv_head_dim(hf_config) -> int:
        return getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

    def _kv_block_bytes(self, hf_config) -> int:
        """Bytes one KV-cache block costs for a model with this config."""
        return (2 * hf_config.num_hidden_layers * self.config.kv_cache_block_size
                * hf_config.num_key_value_heads * self._kv_head_dim(hf_config)
                * hf_config.dtype.itemsize)

    def _alloc_kv_cache(self, hf_config, num_blocks: int) -> torch.Tensor:
        return torch.empty(2, hf_config.num_hidden_layers, num_blocks,
                           self.config.kv_cache_block_size,
                           hf_config.num_key_value_heads, self._kv_head_dim(hf_config))

    @staticmethod
    def _wire_kv_cache(model, kv_cache: torch.Tensor) -> int:
        """Point each attention layer at its slice of `kv_cache`."""
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        return layer_id

    def _init_kv_cache(self):
        logging.info("Initializing key-value cache...")

        config = self.config
        hf_config = self.config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        block_bytes = self._kv_block_bytes(hf_config)
        if self.draft_model is not None:
            # The draft cache is allocated with the same block size and the
            # same number of blocks as the target cache, so one block id costs
            # a block in each and both have to come out of the same budget.
            block_bytes += self._kv_block_bytes(config.draft_hf_config)

        budget = int(total * config.gpu_memory_utilization - used - peak + current)
        kv_cache_num_blocks = budget // block_bytes
        assert kv_cache_num_blocks > 0, (
            f"No memory left for the KV cache: budget is {budget / 2**20:.0f} MiB "
            f"(gpu_memory_utilization={config.gpu_memory_utilization} of "
            f"{total / 2**20:.0f} MiB total, minus {used / 2**20:.0f} MiB already used "
            f"and a {peak / 2**20:.0f} MiB warmup peak), but one block costs "
            f"{block_bytes / 2**20:.1f} MiB. Raise gpu_memory_utilization, or lower "
            f"kv_cache_block_size / max_num_batched_tokens / max_model_len."
        )

        kv_cache_num_blocks = min(kv_cache_num_blocks, config.kv_cache_num_blocks)

        # update the config with the new value
        config.kv_cache_num_blocks = kv_cache_num_blocks

        self.kv_cache = self._alloc_kv_cache(hf_config, kv_cache_num_blocks)
        num_wired = self._wire_kv_cache(self.model, self.kv_cache)
        assert num_wired == hf_config.num_hidden_layers

        logging.info(f'Allocated {kv_cache_num_blocks} key-value cache blocks.')

        if self.draft_model is not None:
            # The draft gets its own cache tensor but *not* its own allocator:
            # a slot index is block_id * kv_cache_block_size + offset, which
            # depends only on the block size, so the block ids in req.blocks —
            # handed out by the single KVCacheBlockManager — address both
            # caches. ctx.slot_mapping and ctx.block_table are therefore valid
            # for the draft as-is, and the block manager needs no changes.
            #
            # This deviates from speculative-decoding-plan.md, which called for
            # a non-paged contiguous draft cache. FlashAttention.forward has no
            # non-paged path, so that would have meant a second attention path
            # in a file the target model also depends on. See PROGRESS.md.
            self.draft_kv_cache = self._alloc_kv_cache(config.draft_hf_config, kv_cache_num_blocks)
            num_wired = self._wire_kv_cache(self.draft_model, self.draft_kv_cache)
            assert num_wired == config.draft_hf_config.num_hidden_layers

            logging.info(f'Allocated a draft key-value cache over the same {kv_cache_num_blocks} blocks.')

    def _warmup_model(self):
        logging.info("Warming up model...")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_batched_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_batched_seqs)

        # for fast start
        max_model_len = 64
        num_batched_seqs = 10
        reqs = [Request([0] * max_model_len) for _ in range(num_batched_seqs)]
        
        self.execute(Batch(Batch.PREFILL, reqs))
        
        torch.cuda.empty_cache()


    def _build_prefill_input(self, requests: list[Request]) -> tuple[torch.Tensor, Context]:
        input_ids = []
        positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        block_table = None

        for req in requests:
            seqlen = len(req.tokens)
            input_ids.extend(req.tokens[req.num_cached_tokens:])
            positions.extend(range(req.num_cached_tokens, seqlen))
            
            seqlen_q = seqlen - req.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            num_cached_blocks = req.num_cached_tokens // self.config.kv_cache_block_size
            for i in range(num_cached_blocks, len(req.blocks)):
                start = req.blocks[i] * self.config.kv_cache_block_size
                if i != len(req.blocks) - 1:
                    end = start + self.config.kv_cache_block_size
                else:
                    last_block_tokens = len(req.tokens) - (len(req.blocks) - 1) * self.config.kv_cache_block_size
                    end = start + last_block_tokens
                slot_mapping.extend(list(range(start, end)))
                
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_table = self._build_block_table(requests)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx = Context(
            prefill=True,
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q = max_seqlen_q,
            max_seqlen_k = max_seqlen_k,
            block_table = block_table
        )
        return input_ids, ctx


    def _build_decode_input(self, requests: list[Request]) -> tuple[torch.Tensor, Context]:
        input_ids = []
        positions = []
        slot_mapping = []
        cache_seqlens = []

        for req in requests:
            input_ids.append(req.tokens[-1])
            positions.append(len(req.tokens) - 1)
            cache_seqlens.append(len(req.tokens))

            slot_base_index = req.blocks[-1] * self.config.kv_cache_block_size
            last_block_tokens = len(req.tokens) - (len(req.blocks) - 1) * self.config.kv_cache_block_size
            slot_mapping.append(slot_base_index + last_block_tokens - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx = Context(
            prefill=False,
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            block_table = self._build_block_table(requests),
        )
        return input_ids, ctx


    def _build_verify_input(self, requests: list[Request],
                            proposals: list[list[int]]) -> tuple[torch.Tensor, Context]:
        """Inputs for a speculative verify pass: K+1 query tokens per request.

        Entering a round, a request's KV cache holds `req.tokens[:-1]` — the
        last committed token has been sampled but never fed through the model
        (this is the same invariant the Phase 2 prototype maintains, and the
        reason `_build_decode_input` feeds `req.tokens[-1]`). The verify pass
        feeds that token followed by the draft's K proposals, so query token j
        sits at absolute position `len(req.tokens) - 1 + j` and must attend to
        keys 0..that position — which includes the proposals ahead of it in the
        same pass, already written to the cache by `store_kvcache`.

        Shape note, and a correction to the plan. This is *not* routed through
        `flash_attn_varlen_func`, which speculative-decoding-plan.md section 7
        and the Phase 0 note both assumed it would be. That kernel anchors its
        causal mask TOP-LEFT: a K+1-token query segment against a longer key
        sequence attends to keys 0..j rather than to its own prefix, leaving
        every query blind to the request's history. Measured, with the kernel
        source cited, in experiments/paged_varlen_check.py.

        The decode kernel does express this shape. Its `seqlen_q == 1` assert
        constrains the *query* dimension, not the number of tokens in flight,
        so the K+1 tokens ride in the batch dimension as K+1 rows, each with
        its own `cache_seqlens` and its own copy of the request's block-table
        row. The result is still one forward pass over
        `num_requests * (K+1)` tokens with one GEMM per projection; only the
        attention gather is per-row.
        """
        block_size = self.config.kv_cache_block_size

        input_ids = []
        positions = []
        slot_mapping = []
        cache_seqlens = []
        block_table = []

        max_block_len = max(len(req.blocks) for req in requests)

        for req, proposal in zip(requests, proposals):
            base = len(req.tokens) - 1
            padded_blocks = req.blocks + [-1] * (max_block_len - len(req.blocks))

            for offset, token in enumerate([req.tokens[-1], *proposal]):
                pos = base + offset
                assert pos // block_size < len(req.blocks), (
                    f"request {req.id} holds {len(req.blocks)} blocks, too few to reach "
                    f"position {pos}. A verify pass writes KV for tokens that are not in "
                    f"req.tokens yet, so the caller must reserve them first with "
                    f"allocate_block_for_decode(req, extra_tokens={len(proposal)})."
                )
                input_ids.append(token)
                positions.append(pos)
                # Each row attends to its own prefix, inclusive of itself.
                cache_seqlens.append(pos + 1)
                slot_mapping.append(req.blocks[pos // block_size] * block_size + pos % block_size)
                block_table.append(padded_blocks)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx = Context(
            prefill=False,
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            block_table = torch.tensor(block_table, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
        )
        return input_ids, ctx

    @torch.inference_mode()
    def verify(self, requests: list[Request], proposals: list[list[int]]) -> torch.Tensor:
        """Score the draft's proposals against the target model in one pass.

        Returns `(num_requests, K+1, vocab_size)`. Row j of a request is the
        target's distribution for the token that *follows* query token j: row 0
        follows the request's last committed token — so it is exactly what a
        plain decode step would have produced — row j>0 follows proposal j-1,
        and row K is the bonus distribution used when every proposal is
        accepted.

        **Stale KV.** The pass writes cache entries for all K proposals before
        anyone knows how many will be accepted. If only M < K are, the slots for
        positions [len+M, len+K) hold KV for tokens that never entered the
        sequence. They are deliberately left in place rather than zeroed,
        because nothing can read them: every read is bounded by `cache_seqlens`,
        which is derived from the request's committed token count, and the next
        round's verify pass begins writing at exactly the first dead slot. So
        the stale region is unreadable until it is overwritten, and clearing it
        would be pure cost. This is the "cache_seqlens is the source of truth"
        option from speculative-decoding-plan.md section 7.3. The zero-acceptance
        case (M=0), where the whole proposal region goes stale every round, is
        the one most implementations get wrong and is covered explicitly by
        experiments/verify_pass_gate.py.

        **CUDA graphs are bypassed.** `CudaGraphRunner` captures one query row
        per request and a verify pass has K+1, so this calls the model eagerly.
        Recorded as a known limitation, per plan section 7.4.
        """
        assert requests, "verify() needs at least one request"
        num_proposals = len(proposals[0])
        assert all(len(p) == num_proposals for p in proposals), (
            "every request in a verify batch must carry the same number of proposals, "
            "since the returned logits are a dense (num_requests, K+1, vocab) tensor"
        )

        input_ids, ctx = self._build_verify_input(requests, proposals)
        logits = self.model(ctx, input_ids, ctx.positions)
        return logits.view(len(requests), num_proposals + 1, -1)

    @staticmethod
    def _build_block_table(requests: list[Request]) -> torch.Tensor:
        max_block_len = max(len(req.blocks) for req in requests)
        block_table = [
            req.blocks + [-1] * (max_block_len - len(req.blocks))
            for req in requests
        ]
        return torch.tensor(block_table, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def _build_sample_input(self, requests: list[Request]) -> tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None]:
        temperatures = []
        top_ks = []
        top_ps = []
        top_p_k_needed = False
        temperatures_needed = False
        for req in requests:
            temperatures.append(req.sampling_params.temperature)
            top_ks.append(req.sampling_params.top_k)
            top_ps.append(req.sampling_params.top_p)
            if req.sampling_params.top_k > 0 or req.sampling_params.top_p < 1.0:
                top_p_k_needed = True
            if req.sampling_params.temperature != 1.0:
                temperatures_needed = True
        
        top_ks_tensor = None
        top_ps_tensor = None
        temperatures_tensor = None
        
        if top_p_k_needed:
            top_ks_tensor = torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            top_ps_tensor = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
            
        if temperatures_needed:
            temperatures_tensor = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    
        return temperatures_tensor, top_ks_tensor, top_ps_tensor
    
    def sample(self, logits: torch.Tensor, batch: Batch) -> list[int]:
        temperatures, top_k, top_p = self._build_sample_input(batch.requests)
        return self.sampler(logits, temperatures, top_k, top_p).tolist()

    def prefill(self, ctx: Context, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(ctx, input_ids, ctx.positions)
        return logits

    def decode(self, ctx: Context, input_ids: torch.Tensor) -> torch.Tensor:
        if self.config.use_cuda_graph and self.graph_runner.max_batch_size >= input_ids.size(0):
            logits = self.graph_runner.replay(ctx, input_ids)
        else:
            logits = self.model(ctx, input_ids, ctx.positions)
        return logits
    
    def forward(self, ctx: Context, tokens: torch.Tensor) -> torch.Tensor:
        if ctx.prefill:
            logits = self.prefill(ctx, tokens)
        else:
            logits = self.decode(ctx, tokens)
            
        return logits

    @torch.inference_mode()
    def execute(self, batch: Batch) -> list[int]:
        if batch.type == Batch.PREFILL:
            input_ids, ctx = self._build_prefill_input(batch.requests)
        else:
            input_ids, ctx = self._build_decode_input(batch.requests)

        logits = self.forward(ctx, input_ids)
        output_tokens = self.sample(logits, batch)
        return output_tokens
    