import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                logits: torch.Tensor,
                temperatures: torch.Tensor|None,
                top_p: torch.Tensor|None,
                top_k: torch.Tensor|None) -> torch.Tensor:

        if temperatures is not None:
            logits = logits / temperatures.unsqueeze(-1)
        
        if top_p is not None or top_k is not None:
            # imported lazily: flashinfer is only needed for top-k/top-p, it is
            # absent from requirements.txt, and it has no reliable Windows
            # wheel. Greedy decoding — which the speculative-decoding
            # correctness harness runs on — must not depend on it.
            from flashinfer.sampling import top_k_top_p_sampling_from_logits

            # flashinfer has a bug when default device is cuda
            # see https://github.com/flashinfer-ai/flashinfer/issues/2333
            device = torch.get_default_device()
            torch.set_default_device("cpu")
            sampled_tokens = top_k_top_p_sampling_from_logits(
                logits,
                top_k,
                top_p,
            )
            torch.set_default_device(device)
            return sampled_tokens
        
        return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # speculative decoding — a second sampling path, kept off `forward`
    # ------------------------------------------------------------------
    #
    # `forward` above is the plain-decode path and must stay untouched: it is
    # what every non-speculative step runs, and what speculative decoding has
    # to reproduce exactly.

    @staticmethod
    def greedy_probs(logits: torch.Tensor) -> torch.Tensor:
        """Rows of `logits` -> one-hot probability rows on their argmax.

        Temperature 0 is represented as a distribution rather than as a branch
        inside `rejection_sample`, so greedy runs through the identical
        rejection-sampling code path as any other temperature. That is what
        makes the greedy exact-match harness a real test of the general path
        instead of a test of a shortcut, and it is the same choice the Phase 2
        prototype made (experiments/spec_decode_prototype.py).

        Note this must mirror what `forward` does, not what the sampling
        parameters nominally say: `forward` takes its argmax branch whenever
        top-k and top-p are unset, *whatever* the temperature, because scaling
        logits by a positive constant cannot move the argmax.
        """
        probs = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device)
        probs.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return probs

    def rejection_sample(self,
                         draft_tokens: torch.Tensor,
                         q: torch.Tensor,
                         p: torch.Tensor,
                         generator: torch.Generator | None = None,
                         ) -> list[tuple[list[int], int]]:
        """Modified rejection sampling (Leviathan et al. 2023, Chen et al. 2023).

        Ported from `rejection_sample` in experiments/spec_decode_prototype.py,
        which was written as a pure function for exactly this move and whose
        output distribution is gated there against the target's at 40k trials.
        The only change is that this operates on a batch of requests at once.

        Args:
            draft_tokens: (B, K) int64 — the tokens the draft proposed.
            q: (B, K, V) — the draft's distribution at each proposal step. Row i
               is the distribution `draft_tokens[:, i]` was actually drawn from.
            p: (B, K+1, V) — the target's distribution at those same K positions
               plus a (K+1)-th row for the bonus token that follows a full
               acceptance.
            generator: RNG, passed explicitly so a run is reproducible. Must
               live on the same device as `p`.

        Returns:
            One `(tokens, num_accepted)` per request. `tokens` is the accepted
            prefix of that request's proposals followed by exactly one more
            token — the residual resample on rejection, or the bonus token on
            full acceptance — so `len(tokens) == num_accepted + 1` always. A
            round can therefore never return zero tokens, which is what
            guarantees forward progress when the draft is useless.

        The whole batch's accept/reject decisions are made in one shot and moved
        to the host together. Reading `p[b, i, x]` one scalar at a time, as the
        prototype does, costs a device sync per proposal; at B*K of those per
        round it would dominate a step on a small model.
        """
        num_requests, num_draft = draft_tokens.shape
        assert q.shape[:2] == (num_requests, num_draft), f"q has shape {tuple(q.shape)}"
        assert p.shape[:2] == (num_requests, num_draft + 1), f"p has shape {tuple(p.shape)}"

        # p(x) and q(x) for the proposed token x, at every (request, step).
        index = draft_tokens.unsqueeze(-1)
        q_x = q.gather(-1, index).squeeze(-1)
        p_x = p[:, :num_draft].gather(-1, index).squeeze(-1)

        # accept with probability min(1, p(x)/q(x)). q(x) == 0 is unreachable —
        # x was drawn from q — but a numerically-zeroed row must not produce a
        # nan ratio, so it is forced to an accept exactly as the prototype does.
        nonzero = q_x > 0
        ratio = torch.where(nonzero, p_x / torch.where(nonzero, q_x, torch.ones_like(q_x)),
                            torch.ones_like(p_x))
        uniforms = torch.rand(p_x.shape, generator=generator, device=p.device)
        accepted = (uniforms < ratio.clamp(max=1.0)).tolist()
        proposed = draft_tokens.tolist()

        # How many leading proposals each request keeps.
        num_accepted = []
        for row in accepted:
            count = num_draft
            for i, ok in enumerate(row):
                if not ok:
                    count = i
                    break
            num_accepted.append(count)

        # The one extra token per request: on rejection at step m it comes from
        # the renormalised residual max(0, p - q) at that step, and on full
        # acceptance from the target's bonus row. Built as a (B, V) matrix so
        # the draw is a single multinomial call.
        finals = []
        for b, m in enumerate(num_accepted):
            if m == num_draft:
                finals.append(p[b, num_draft])
                continue
            residual = torch.clamp(p[b, m] - q[b, m], min=0.0)
            # Reachable only through floating-point noise: a rejection implies
            # p != q somewhere, hence positive residual mass. Selected on device
            # to avoid a sync per request.
            residual = torch.where(residual.sum() > 0, residual, p[b, m])
            finals.append(residual / residual.sum())

        extra = torch.multinomial(torch.stack(finals), 1, generator=generator).squeeze(-1).tolist()

        return [(proposed[b][:m] + [extra[b]], m) for b, m in enumerate(num_accepted)]
