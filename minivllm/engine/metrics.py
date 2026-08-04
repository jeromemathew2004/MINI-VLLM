import time
from minivllm.scheduler.batch import Batch

class Stats:
    def __init__(self):
        self.prefill_throughput = 0
        self.decode_throughput = 0
        self.time_to_first_token = 0
        self.inter_token_latency = 0
        self.tokens_per_second = 0
        self.requests_per_second = 0
        self.finished_requests = 0

        # Speculative decoding. Both stay 0 when it is off.
        self.acceptance_rate = 0
        # Tokens one request gets out of one decode step. Exactly 1.0 without
        # speculation, and between 1.0 and K+1 with it — deliberately not
        # tokens-per-batch-step, which would fold in the batch width and hide
        # the thing being measured.
        self.tokens_per_request_step = 0
        # Fraction of decode steps that actually ran a round. Always 1.0 with a
        # draft model, which proposes unconditionally, and below it with the
        # n-gram proposer, which declines when nothing in the batch matched.
        # Read it alongside acceptance_rate: a high acceptance rate over a
        # handful of rounds is a different claim from one over every step.
        self.speculation_rate = 0

class Metrics:
    def __init__(self):
        self.prefill_time = 0
        self.prefill_steps = 0
        self.prefill_tokens = 0

        self.decode_time = 0
        self.decode_tokens = 0
        self.decode_steps = 0
        # Requests summed over decode steps. decode_steps counts batches, so it
        # is the wrong denominator for "tokens one request got out of one step".
        self.decode_request_steps = 0

        # Speculative decoding: tokens the draft offered, and how many the
        # target's rejection sampler kept. Acceptance rate is the number that
        # explains whether a given K pays for itself, which is what Phase 7's
        # sweep charts.
        self.spec_proposed = 0
        self.spec_accepted = 0
        # Decode steps that ran a round, as opposed to falling back to a plain
        # decode step because the proposer had nothing (n-gram only).
        self.spec_rounds = 0

        self.finished_request_count = 0
        self.start_time = time.perf_counter()

    def update(self, batch: Batch, committed: list[int] | None = None,
               num_proposed: int = 0, num_accepted: int = 0):
        """Record one step.

        `committed` is the per-request token count the scheduler actually
        appended. A plain decode step commits one per request, so leaving it
        None keeps the old behaviour; a speculative round commits between 1 and
        K+1 and cannot be inferred from the batch width.
        """
        if batch.type == Batch.PREFILL:
            self._update_prefill_metrics(batch)
        else:
            self._update_decode_metrics(batch, committed, num_proposed, num_accepted)

    def _update_prefill_metrics(self, batch: Batch):
        self.prefill_steps += 1
        self.prefill_time += time.perf_counter() - batch.create_time

        for req in batch.requests:
            self.prefill_tokens += len(req.tokens) - req.num_cached_tokens

    def _update_decode_metrics(self, batch: Batch, committed: list[int] | None,
                               num_proposed: int, num_accepted: int):
        self.decode_steps += 1
        self.decode_request_steps += len(batch.requests)
        self.decode_tokens += sum(committed) if committed is not None else len(batch.requests)
        self.decode_time += time.perf_counter() - batch.create_time

        self.spec_proposed += num_proposed
        self.spec_accepted += num_accepted
        if num_proposed > 0:
            self.spec_rounds += 1

        for req in batch.requests:
            if req.finished:
                self.finished_request_count += 1

    def reset(self):
        self.prefill_time = 0
        self.decode_time = 0
        self.prefill_steps = 0
        self.decode_steps = 0
        self.decode_request_steps = 0
        self.spec_proposed = 0
        self.spec_accepted = 0
        self.spec_rounds = 0
        self.finished_request_count = 0
        self.start_time = time.perf_counter()


    def stats(self) -> Stats:
        tokens = self.prefill_tokens + self.decode_tokens
        elapsed = time.perf_counter() - self.start_time

        stats = Stats()
        stats.time_to_first_token = self.prefill_time / self.prefill_steps if self.prefill_steps > 0 else 0
        stats.inter_token_latency = self.decode_time / self.decode_steps if self.decode_steps > 0 else 0
        stats.tokens_per_second = tokens / elapsed if elapsed > 0 else 0
        stats.requests_per_second = self.finished_request_count / elapsed if elapsed > 0 else 0
        stats.prefill_throughput = self.prefill_tokens / self.prefill_time if self.prefill_time > 0 else 0
        stats.decode_throughput = self.decode_tokens / self.decode_time if self.decode_time > 0 else 0
        stats.finished_requests = self.finished_request_count
        stats.acceptance_rate = self.spec_accepted / self.spec_proposed if self.spec_proposed > 0 else 0
        stats.tokens_per_request_step = (self.decode_tokens / self.decode_request_steps
                                         if self.decode_request_steps > 0 else 0)
        stats.speculation_rate = (self.spec_rounds / self.decode_steps
                                  if self.decode_steps > 0 else 0)
        return stats
