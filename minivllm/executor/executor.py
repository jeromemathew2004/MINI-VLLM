import torch
import logging

from minivllm.config.config import Config
from minivllm.engine.request import Request
from minivllm.executor import ngram
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
        # _init_kv_cache reads already account for the draft's weights. Stays
        # None under speculative_method="ngram", which proposes from the
        # request's own token history — that is most of why the n-gram round is
        # cheaper: no second model to run, and ~50 KV-cache blocks handed back.
        self.draft_model = (load_draft_model(config)
                            if config.use_speculative_decoding
                            and config.speculative_method == "draft" else None)
        self.draft_kv_cache = None

        # RNG for rejection sampling. None means torch's default generator;
        # tests set it to make a round reproducible. Under greedy sampling the
        # round is deterministic regardless — p and q are one-hot, so every
        # accept/reject test and every residual draw has a single outcome.
        self.spec_generator: torch.Generator | None = None

        self.sampler = Sampler()

        self._warmup_model()

        self._init_kv_cache()
        
        self.graph_runner = None
        self.draft_graph_runner = None
        if config.use_cuda_graph:
            logging.info("Initializing CUDA graph executor...")

            # A speculative round is decode-shaped — that is Phase 4's finding:
            # a request's K+1 tokens ride in the *batch* dimension, so a verify
            # pass over B requests is a decode call at width B*(K+1) and needs
            # graphs captured that wide.
            #
            # This sizing is the difference between the feature being usable and
            # not. Measured on the dev host, graphs are worth ~8x per decode
            # step; an eager round costs ~15x a graphed step, which no
            # acceptance rate can repay, since break-even would need more
            # accepted tokens than a round even proposes. See
            # experiments/spec_breakeven.py.
            width = config.max_num_batched_seqs
            if config.use_speculative_decoding:
                width *= config.num_speculative_tokens + 1

            self.graph_runner = CudaGraphRunner(self.model, config, width, label="target")
            self.graph_runner.capture()

            if self.draft_model is not None:
                # The draft contributes one row per request per proposal step,
                # and two on the first (the catch-up row; see propose()).
                self.draft_graph_runner = CudaGraphRunner(
                    self.draft_model, config, 2 * config.max_num_batched_seqs,
                    hf_config=config.draft_hf_config, label="draft")
                self.draft_graph_runner.capture()

        
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

        block_size = self.config.kv_cache_block_size
        for req in requests:
            pos = len(req.tokens) - 1
            input_ids.append(req.tokens[-1])
            positions.append(pos)
            cache_seqlens.append(len(req.tokens))

            # Resolve the slot from the token's position rather than from
            # req.blocks[-1]. The two agree whenever a request holds exactly
            # cdiv(len, block_size) blocks, which is every non-speculative case,
            # but a speculative round reserves K tokens of slack: the last block
            # is then one no committed token lives in yet, and the old
            # arithmetic produced a negative offset into it.
            slot_mapping.append(req.blocks[pos // block_size] * block_size + pos % block_size)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx = Context(
            prefill=False,
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            block_table = self._build_block_table(requests),
        )
        return input_ids, ctx


    def _build_paged_rows(self, requests: list[Request],
                          rows: list[list[tuple[int, int]]]) -> tuple[torch.Tensor, Context]:
        """Decode-shaped inputs from an explicit (token, position) list per request.

        Every row is one query token: it is written into the paged cache at the
        slot its absolute position maps to, and then attends to keys
        `0..position` of its own request. `_build_decode_input` is the
        one-row-per-request special case of this; a verify pass and the draft's
        proposal steps pass more than one.

        The extra rows of a request ride in the *batch* dimension rather than in
        a query dimension — see `_build_verify_input` for why that is the only
        multi-token shape mini-flash-attention can express.
        """
        block_size = self.config.kv_cache_block_size

        input_ids = []
        positions = []
        slot_mapping = []
        cache_seqlens = []
        block_table = []

        max_block_len = max(len(req.blocks) for req in requests)

        for req, req_rows in zip(requests, rows):
            padded_blocks = req.blocks + [-1] * (max_block_len - len(req.blocks))

            for token, pos in req_rows:
                assert pos // block_size < len(req.blocks), (
                    f"request {req.id} holds {len(req.blocks)} blocks, too few to reach "
                    f"position {pos}. A speculative round writes KV for tokens that are "
                    f"not in req.tokens yet, so the caller must reserve them first with "
                    f"allocate_block_for_decode(req, extra_tokens=K)."
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
        rows = []
        for req, proposal in zip(requests, proposals):
            base = len(req.tokens) - 1
            rows.append([(token, base + offset)
                         for offset, token in enumerate([req.tokens[-1], *proposal])])
        return self._build_paged_rows(requests, rows)

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

        **CUDA graphs.** A verify pass replays the target's captured graphs like
        any other decode-shaped call — "one query row per request" was never the
        constraint, "one query row per *batch entry*" is, and this shape
        satisfies it at width `num_requests * (K+1)`. `Executor.__init__` sizes
        the runner accordingly. Without this the pass runs ~8x slower and
        speculation cannot break even at any acceptance rate; see
        experiments/spec_breakeven.py.
        """
        assert requests, "verify() needs at least one request"
        num_proposals = len(proposals[0])
        assert all(len(p) == num_proposals for p in proposals), (
            "every request in a verify batch must carry the same number of proposals, "
            "since the returned logits are a dense (num_requests, K+1, vocab) tensor"
        )

        input_ids, ctx = self._build_verify_input(requests, proposals)
        logits = self._decode_forward(self.model, self.graph_runner, ctx, input_ids)
        return logits.view(len(requests), num_proposals + 1, -1)

    @torch.inference_mode()
    def propose(self, requests: list[Request]) -> tuple[list[list[int]], torch.Tensor]:
        """Run the draft model K times to propose K tokens per request.

        Returns `(proposals, q)`, where `proposals[i]` is request i's K token
        ids and `q` is the `(num_requests, K, vocab_size)` distributions they
        were drawn from. Rejection sampling needs the distribution and not just
        the token, since it compares `q(x)` against the target's `p(x)`.

        The draft is run greedily, matching the target — `execute_speculative`
        rejects any other sampling policy.

        **Why the first step feeds two tokens.** The draft shares the target's
        block ids (Phase 3) but keeps its own position frontier, and it ends a
        round one token behind whenever every proposal was accepted: it runs K
        steps covering positions `base .. base+K-1`, while a full acceptance
        commits K+1 tokens. Rather than track a cached length per request and
        assemble a ragged catch-up batch, each round simply re-feeds the
        second-to-last committed token alongside the last one, which closes a
        deficit that is provably never larger than 1.

        Re-feeding a position is unconditionally safe, not merely safe when the
        draft is behind. The K/V a layer computes for position i depends only on
        the tokens at `0..i`, all of which are already correct in the cache, so
        on the rounds where the draft was not behind the extra row rewrites the
        identical bytes. The cost is one extra row per request through a 42M
        parameter model.

        Note a stale draft cache would cost acceptance rate and never
        correctness: rejection sampling returns the target's distribution for
        *any* q, provided q is the distribution the token was actually drawn
        from — which it is here by construction, whatever the cache held.
        """
        assert self.draft_model is not None, (
            "propose() needs a draft model; construct the engine with "
            "use_speculative_decoding=True"
        )
        num_speculative = self.config.num_speculative_tokens

        proposals: list[list[int]] = [[] for _ in requests]
        q_rows: list[torch.Tensor] = []

        for step in range(num_speculative):
            rows = []
            for req, proposal in zip(requests, proposals):
                base = len(req.tokens) - 1
                if step == 0:
                    assert len(req.tokens) >= 2, (
                        f"request {req.id} holds {len(req.tokens)} token(s); a decode "
                        f"step only runs once prefill has emitted one, so there are "
                        f"always at least two"
                    )
                    rows.append([(req.tokens[-2], base - 1), (req.tokens[-1], base)])
                else:
                    rows.append([(proposal[-1], base + step)])

            input_ids, ctx = self._build_paged_rows(requests, rows)
            logits = self._decode_forward(self.draft_model, self.draft_graph_runner,
                                          ctx, input_ids)

            if step == 0:
                # Two rows per request; only the second one predicts a new token.
                logits = logits.view(len(requests), 2, -1)[:, 1]

            probs = self.sampler.greedy_probs(logits)
            q_rows.append(probs)
            for i, token in enumerate(probs.argmax(-1).tolist()):
                proposals[i].append(token)

        return proposals, torch.stack(q_rows, dim=1)

    def propose_ngram(self, requests: list[Request]
                      ) -> tuple[list[list[int]] | None, torch.Tensor | None]:
        """Propose by looking the tail of each request up in its own history.

        Same contract as `propose`: `(proposals, q)` with `q` of shape
        `(num_requests, K, vocab_size)`. A lookup is deterministic, so `q` is
        one-hot on the proposed token — the same thing a greedy draft returns,
        which is why nothing downstream of here needs to know which proposer
        ran. See `minivllm/executor/ngram.py`.

        **Returns `(None, None)` when no request in the batch has a match**, and
        the caller must then fall back to a plain decode step. This is not an
        optimisation, it is what keeps the method from being a pessimisation:
        measured on the dev host a round costs ~12 ms against ~9 ms for a
        graphed decode step, so a round that can only ever emit one token loses
        ~40%. On open-ended text the lookup misses most of the time, and
        skipping is what bounds the damage to zero.

        A batch can still be mixed — some requests matched, some did not. Those
        get filler proposals so the batch stays rectangular (`verify` returns a
        dense tensor and asserts uniform proposal length); filler is rejected at
        i=0 and the round emits their one token, exactly as a plain step would.
        """
        num_speculative = self.config.num_speculative_tokens
        max_match_len = self.config.ngram_max_match_len
        min_match_len = self.config.ngram_min_match_len

        proposals: list[list[int]] = []
        any_match = False

        for req in requests:
            proposal = ngram.lookup(req.tokens, num_speculative, max_match_len,
                                    min_match_len)
            if proposal is None:
                # Filler: repeating the last committed token is in-vocabulary
                # and needs no magic constant. It is rejected unless the target
                # itself wanted to repeat, in which case accepting it was right.
                proposals.append([req.tokens[-1]] * num_speculative)
            else:
                any_match = True
                proposals.append(proposal)

        if not any_match:
            return None, None

        draft_tokens = torch.tensor(proposals, dtype=torch.int64, device="cuda")
        q = torch.zeros(len(requests), num_speculative, self.config.hf_config.vocab_size,
                        dtype=torch.float32, device="cuda")
        q.scatter_(-1, draft_tokens.unsqueeze(-1), 1.0)
        return proposals, q

    def _propose(self, requests: list[Request]
                 ) -> tuple[list[list[int]] | None, torch.Tensor | None]:
        """Dispatch to the configured proposer. `(None, None)` means "skip"."""
        if self.config.speculative_method == "ngram":
            return self.propose_ngram(requests)
        return self.propose(requests)

    @torch.inference_mode()
    def execute_speculative(self, batch: Batch) -> tuple[list[list[int]], list[int], int]:
        """One speculative round: propose -> verify -> rejection-sample.

        Returns `(tokens, num_accepted, num_proposed)`. `tokens[i]` is between 1
        and K+1 token ids for request i and `num_accepted[i]` is how many of its
        K proposals survived; committing them is the scheduler's job, so this
        method leaves `req.tokens` alone.

        `num_proposed` is `len(requests) * K` for a round that ran and **0 for a
        step that degenerated into a plain decode** because the proposer had
        nothing to offer — only the n-gram proposer can do that, and only when
        no request in the batch matched. It is returned rather than recomputed
        by the caller so the acceptance rate stays an honest ratio: a skipped
        step proposed nothing and accepted nothing, and folding a phantom K
        proposals into the denominator would understate the proposer whenever it
        is being correctly conservative.

        The caller must already have reserved K tokens of slack per request with
        `allocate_block_for_decode(req, extra_tokens=K)`, because both the draft
        steps and the verify pass write KV past the last committed token.
        """
        requests = batch.requests
        for req in requests:
            sampling_params = req.sampling_params
            assert sampling_params.top_k <= 0 and sampling_params.top_p >= 1.0, (
                f"request {req.id} asks for top-k/top-p sampling, which the speculative "
                f"path does not implement. Rejection sampling has to compare the exact "
                f"distribution the non-speculative path would have sampled from, and for "
                f"top-k/top-p that distribution lives inside flashinfer's fused kernel, "
                f"which never exposes it. Greedy (top_k=0, top_p=1.0 — Sampler.forward's "
                f"argmax branch) is what this path reproduces and what every correctness "
                f"gate runs."
            )

        proposals, q = self._propose(requests)

        if proposals is None:
            # Nothing to verify. Fall back to the plain decode path rather than
            # running a round over filler: a round is ~40% more expensive than a
            # graphed decode step, and one that cannot accept anything spends
            # that for a single token. Shape the result like a round's so the
            # caller has one code path.
            tokens = self.execute(batch)
            return [[token] for token in tokens], [0] * len(requests), 0

        target_logits = self.verify(requests, proposals)
        p = self.sampler.greedy_probs(target_logits)

        draft_tokens = torch.tensor(proposals, dtype=torch.int64, device=q.device)
        results = self.sampler.rejection_sample(draft_tokens, q, p, self.spec_generator)
        num_proposed = len(requests) * self.config.num_speculative_tokens
        return ([tokens for tokens, _ in results],
                [num for _, num in results],
                num_proposed)

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

    @staticmethod
    def _decode_forward(model, runner, ctx: Context, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a decode-shaped pass, replaying a captured graph when one fits.

        Plain decode, the draft's proposal steps and the verify pass all take
        this route — they differ only in how many rows each request contributes.
        Falling back to eager when the batch is wider than anything captured
        keeps graph sizing a performance question rather than a correctness one.

        When a graph is used the result is a **view of that runner's persistent
        output buffer**, valid only until the same runner replays again. Every
        caller here either samples from it or converts it to probabilities
        immediately; the target and the draft have separate runners, so a round
        interleaving them is safe.
        """
        if runner is not None and runner.can_replay(input_ids.size(0)):
            return runner.replay(ctx, input_ids)
        return model(ctx, input_ids, ctx.positions)

    def decode(self, ctx: Context, input_ids: torch.Tensor) -> torch.Tensor:
        return self._decode_forward(self.model, self.graph_runner, ctx, input_ids)
    
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

        if batch.type == Batch.PREFILL and self.draft_model is not None:
            # The draft must enter its first proposal step holding KV for the
            # same prefix the target holds, so it prefills alongside the target
            # over the identical Context — legitimate because the two caches are
            # addressed by the same block ids (see _init_kv_cache). Its logits
            # are discarded: a round's first proposal comes from a draft decode
            # step, not from here.
            #
            # This also runs during _warmup_model, which is deliberate — it puts
            # the draft's activation peak into the figures _init_kv_cache sizes
            # the cache against, where Phase 3 left it unaccounted for.
            self.draft_model(ctx, input_ids, ctx.positions)

        output_tokens = self.sample(logits, batch)
        return output_tokens
    