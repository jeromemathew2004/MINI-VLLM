"""Prompt-lookup proposal: speculative decoding without a draft model.

A draft model exists only to guess the target's next few tokens. On text that
repeats itself — summarisation, code editing, RAG, "rewrite this paragraph with
X changed" — a good guess is usually already sitting in the request's own token
history: find where the last few tokens appeared earlier, and propose whatever
followed them that time. That costs no parameters, no GPU time and no KV cache.

This is a production technique, not a shortcut: vLLM ships it as the `[ngram]`
speculative method and HF transformers as `prompt_lookup_num_tokens`.

It drops into this engine without touching the verify-and-accept core because a
lookup is a *deterministic* proposer, so its proposal distribution `q` is
one-hot — exactly what a greedy draft model returns. `Executor.verify`,
`Sampler.rejection_sample` and the scheduler's multi-token commit cannot tell
the two apart, so the correctness argument established in Phases 4-6 carries
over verbatim: output is the target's, whatever the proposer.

**The honest caveat.** This wins on repetitive workloads and does essentially
nothing on open-ended generation, where the last n tokens have simply never
occurred before. It is not a general-purpose accelerator, and a round that
proposes garbage costs more than the plain decode step it replaced — which is
why `Executor` skips the round entirely when no request in the batch has a
match, rather than proposing filler and letting it be rejected.
"""


def lookup(tokens: list[int], num_speculative: int, max_match_len: int,
           min_match_len: int = 1) -> list[int] | None:
    """Propose `num_speculative` tokens by matching the tail of `tokens` earlier.

    Returns exactly `num_speculative` token ids, or None when no earlier
    occurrence of the tail is long enough to be worth proposing from. Longer
    matches are tried first: a 3-token context that has been seen before is far
    stronger evidence than a 1-token one, and among equal-length matches the
    most recent wins, since the nearest repeat is the likeliest to still be in
    the same context.

    **`min_match_len` is a quality floor, and it matters more than it looks.**
    A 1-token match exists almost everywhere in text of any length — every
    repeated comma is one — so a floor of 1 means the proposer effectively never
    declines, and the round runs unconditionally on evidence that is worth
    roughly nothing. Since a round costs ~1.4x a graphed decode step, proposing
    from weak matches is a *loss*, not a free lottery ticket. Refusing to
    propose is the safe default and the floor is what makes refusing possible.

    The returned list is always full length even when the match ran off the end
    of the history, because `Executor.verify` returns a dense
    `(batch, K+1, vocab)` tensor and asserts every request carries the same
    number of proposals. The short case is padded by repeating the last real
    proposal; that padding is a placeholder expected to be rejected, and
    rejecting it costs nothing beyond the tokens already accepted before it.
    """
    assert num_speculative >= 1
    assert 1 <= min_match_len <= max_match_len

    n = len(tokens)

    # A match must end before the final token, so the longest pattern that can
    # possibly have an earlier occurrence is n - 1 tokens long.
    for size in range(min(max_match_len, n - 1), min_match_len - 1, -1):
        pattern = tokens[-size:]
        anchor = pattern[-1]

        # `j` indexes the last token of a candidate match. Scanning downwards
        # takes the most recent match first; stopping at n - 2 keeps the match
        # strictly earlier than the tail itself, which also guarantees at least
        # one token follows it. The cheap `anchor` test keeps the slice
        # comparison off the hot path — most positions fail on one integer.
        for j in range(n - 2, size - 2, -1):
            if tokens[j] != anchor:
                continue
            if tokens[j - size + 1:j + 1] != pattern:
                continue

            proposal = tokens[j + 1:j + 1 + num_speculative]
            return proposal + [proposal[-1]] * (num_speculative - len(proposal))

    return None
