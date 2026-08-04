import logging
from collections import deque

from minivllm.config.config import Config
from minivllm.engine.request import Request
from minivllm.kvcache.block_manager import KVCacheBlockManager
from minivllm.scheduler.batch import Batch

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, config: Config):
        self.max_num_batched_seqs = config.max_num_batched_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_ids = config.eos_token_ids

        self.block_manager = KVCacheBlockManager(config.kv_cache_num_blocks, config.kv_cache_block_size)
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()

        # Slack a decode step needs beyond its one committed token. A
        # speculative round writes KV for K proposed tokens before anyone knows
        # how many will be accepted, so those slots must exist before it runs —
        # and the preemption check below has to know about them too, or the
        # round allocates into a cache that was scheduled as full. Zero when
        # speculation is off, which leaves the decode path exactly as it was.
        self.decode_extra_tokens = (config.num_speculative_tokens
                                    if config.use_speculative_decoding else 0)

        if config.use_speculative_decoding:
            # A speculative round can commit several tokens at once and so may
            # step straight over a block boundary without landing on it, which
            # is the event prefix-cache hashing keys off. The two features are
            # untested together; prefix caching is off by default.
            assert not self.block_manager.support_prefix_cache, (
                "speculative decoding and prefix caching have not been validated "
                "together — a multi-token commit can skip the block boundary that "
                "cache_block_if_needed hashes on, leaving a gap in the prefix chain"
            )


    @property
    def finished(self):
        return not self.waiting and not self.running


    def submit(self, req: Request):
        """
        Add a request to the waiting queue and wait for it to be scheduled.
        """
        self.waiting.append(req)


    def _schedule_prefill(self) -> Batch | None:
        """
        Schedule prefill requests.
        :return: None
        """
        reqs = []
        num_batched_tokens = 0
        while self.waiting and len(reqs) < self.max_num_batched_seqs:
            req = self.waiting[0]
            num_prefill_tokens = len(req.tokens) - req.num_cached_tokens
            if num_batched_tokens + num_prefill_tokens > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate_new_block(req):
                break
            num_batched_tokens += num_prefill_tokens
            self.block_manager.allocate_blocks_for_prefill(req)
            req.state = Request.RUNNING
            self.waiting.popleft()
            self.running.append(req)
            reqs.append(req)
        if reqs:
            return Batch(Batch.PREFILL, reqs)
        return None


    def _schedule_decode(self) -> Batch | None:
        """
        Schedule decode requests.
        :return: None
        """
        reqs = []
        extra = self.decode_extra_tokens
        while self.running and len(reqs) < self.max_num_batched_seqs:
            req = self.running.popleft()
            while not self.block_manager.can_allocate_new_block(req, extra):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(req)
                    break

            if req.state == Request.RUNNING:
                self.block_manager.allocate_block_for_decode(req, extra)
                reqs.append(req)
        if reqs:
            self.running.extendleft(reversed(reqs))
            return Batch(Batch.DECODE, reqs)
        else:
            return None

    def schedule(self) -> Batch | None:
        out = self._schedule_prefill()
        if out is not None:
            return out

        out = self._schedule_decode()
        return out

    def preempt(self, req: Request):
        """
        No more KVCacheBlocks available for the next request, we have to preempt this request
        and release its KVCacheBlocks.
        """
        req.state = Request.WAITING
        self.block_manager.deallocate(req)
        self.waiting.appendleft(req)


    def update(self, batch: Batch, tokens: list[list[int]]) -> list[int]:
        """Commit a step's output tokens to each request.

        `tokens[i]` is a *list* because a speculative round emits between 1 and
        K+1 tokens per request; prefill and plain decode pass a single-element
        list. Returns how many tokens were actually committed per request, which
        is short of what was offered whenever the round ran past the end of the
        sequence — the tokens a round produced after an EOS or after max_tokens
        belong to a continuation that will never be generated, so they are
        dropped rather than emitted. Their KV stays in the cache and is never
        read; the request is finished and its blocks are freed below.
        """
        committed = []
        for req, new_tokens in zip(batch.requests, tokens):
            count = 0
            for token in new_tokens:
                req.append_output_token(token)
                count += 1

                eos_reached = self.eos_token_ids and token in self.eos_token_ids
                max_len_reached = len(req.completion_tokens) >= req.sampling_params.max_tokens
                if  max_len_reached or (eos_reached and not req.sampling_params.ignore_eos):
                    req.state = Request.FINISHED
                    break

            if req.finished:
                self.block_manager.deallocate(req)
                self.running.remove(req)
            else:
                self.block_manager.cache_block_if_needed(req)

            committed.append(count)
        return committed
