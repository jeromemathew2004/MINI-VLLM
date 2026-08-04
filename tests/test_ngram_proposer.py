"""Unit tests for the n-gram (prompt-lookup) proposer.

Pure Python over token lists — no GPU, no checkpoints, no torch. The proposer is
the one part of the speculative path that can be tested exhaustively and
instantly, which is worth exploiting: everything downstream of it is already
covered by the Phase 6 suite and costs minutes per run.

What is *not* tested here, deliberately: whether a proposal is any good. The
correctness of a round does not depend on that (rejection sampling returns the
target's own distribution whatever the proposer offers), and how often the
proposals land is an empirical question about a workload, measured by
`Metrics.acceptance_rate`, not a property of this function.

Usage:
    pytest tests/test_ngram_proposer.py -v
"""

import pytest

from minivllm.executor import ngram

K = 4


def lookup(tokens, k=K, max_match=3, min_match=1):
    return ngram.lookup(tokens, k, max_match, min_match)


# ---------------------------------------------------------------------------
# finding a match
# ---------------------------------------------------------------------------

def test_proposes_the_continuation_of_an_earlier_occurrence():
    # "1 2 3" appeared once and was followed by 4 5 6 7.
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3]
    assert lookup(tokens) == [4, 5, 6, 7]


def test_no_match_returns_none():
    # Strictly increasing: no token, let alone any pair, ever repeats.
    assert lookup(list(range(20))) is None


def test_prefers_the_longest_match():
    # The tail is "9 1 2". A 3-token match exists at index 3 (-> 50), and a
    # shorter "1 2" match also exists at index 8 (-> 99). The longer wins.
    tokens = [7, 7, 7, 9, 1, 2, 50, 51, 52, 53, 1, 2, 60, 61, 62, 63, 9, 1, 2]
    assert lookup(tokens) == [50, 51, 52, 53]


def test_prefers_the_most_recent_among_equal_length_matches():
    # "1 2" occurs at index 0 and again at index 6. The nearer one is likelier
    # to still be in the same context, so it wins.
    tokens = [1, 2, 30, 31, 32, 33, 1, 2, 40, 41, 42, 43, 1, 2]
    assert lookup(tokens) == [40, 41, 42, 43]


def test_falls_back_to_a_shorter_match_when_the_long_one_is_absent():
    # The 3-tail is "5 1 2", which never occurred. "1 2" did.
    tokens = [1, 2, 70, 71, 72, 73, 5, 1, 2]
    assert lookup(tokens) == [70, 71, 72, 73]


# ---------------------------------------------------------------------------
# the shape contract: verify() needs exactly K, always
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_a_hit_always_returns_exactly_k_tokens(k):
    tokens = [1, 2, 3, 4, 5, 1, 2, 3]
    assert len(lookup(tokens, k=k)) == k


def test_a_match_near_the_end_is_padded_rather_than_truncated():
    # The tail "1 2" matches at index 0, and only three tokens follow that match
    # before the history runs out. verify() returns a dense (B, K+1, vocab)
    # tensor and asserts uniform proposal length, so the short case pads.
    tokens = [1, 2, 3, 1, 2]
    proposal = lookup(tokens, k=4)
    assert len(proposal) == 4
    assert proposal[:3] == [3, 1, 2]
    # Padding repeats the last real proposal. It is expected to be rejected;
    # what matters is only that it is a valid token id and that the count is
    # right, since a ragged batch would not survive verify().
    assert proposal[3:] == [2]


def test_the_proposal_may_overlap_the_tail_that_matched():
    # Not a bug: "1 2" was last followed by "3 4 1 2", so proposing that is
    # proposing the text keeps repeating, which is the whole point.
    assert lookup([1, 2, 3, 4, 1, 2], k=4) == [3, 4, 1, 2]


def test_the_match_must_be_strictly_earlier_than_the_tail():
    # A pattern always occurs at the end of its own history. Matching there
    # would propose from tokens that do not exist yet, so the search stops one
    # short — leaving nothing to match against here.
    assert lookup([1, 2], k=2, max_match=2, min_match=1) is None
    # One token of history has no earlier position at all.
    assert lookup([5], k=2, max_match=2, min_match=1) is None


# ---------------------------------------------------------------------------
# the quality floor
# ---------------------------------------------------------------------------

def test_min_match_len_rejects_a_match_that_is_too_short():
    # A single repeated token is the weakest possible evidence, and on real text
    # it is everywhere. With a floor of 2 the proposer declines instead.
    tokens = [1, 9, 9, 9, 2, 3, 4, 1]
    assert lookup(tokens, min_match=1) == [9, 9, 9, 2]
    assert lookup(tokens, min_match=2) is None


def test_min_match_len_still_accepts_a_long_enough_match():
    tokens = [1, 2, 3, 4, 5, 6, 1, 2]
    assert lookup(tokens, min_match=2) == [3, 4, 5, 6]


def test_max_match_len_bounds_the_pattern_but_never_loses_a_match():
    # Raising the cap can only find *stronger* evidence, never fewer matches:
    # the search walks down from the cap to the floor.
    tokens = [4, 5, 1, 2, 3, 90, 91, 92, 93, 1, 2, 3]
    assert lookup(tokens, max_match=1) == [90, 91, 92, 93]
    assert lookup(tokens, max_match=3) == [90, 91, 92, 93]
    assert lookup(tokens, max_match=8) == [90, 91, 92, 93]


# ---------------------------------------------------------------------------
# degenerate inputs — a request is never shorter than 2 tokens at decode time,
# but the function should not depend on the caller for that
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [[], [7], [7, 7, 7]])
def test_short_and_degenerate_histories_do_not_raise(tokens):
    result = lookup(tokens, k=2, max_match=3, min_match=2)
    assert result is None or len(result) == 2


def test_a_fully_repetitive_history_proposes_the_repeat():
    # The pathological-looking case is also the one the method is for: text
    # that repeats itself is exactly where lookup pays.
    #
    # Note the ceiling this exposes. Preferring the *most recent* match means a
    # tail of period p can only ever supply p tokens before the history runs
    # out and padding starts, so K > p buys nothing here. That is the right
    # tradeoff anyway — short-period repetition is a degenerate model loop, not
    # the workload this is for, where the match sits far back in the prompt with
    # plenty of continuation behind it.
    tokens = [1, 2, 3] * 6
    assert lookup(tokens, k=3, min_match=2) == [1, 2, 3]


def test_arguments_are_validated():
    with pytest.raises(AssertionError):
        lookup([1, 2, 3], k=0)
    with pytest.raises(AssertionError):
        # A floor above the cap can never match anything, which is a
        # configuration mistake rather than a "no match" answer.
        lookup([1, 2, 3], max_match=2, min_match=3)


# ---------------------------------------------------------------------------
# the proposer never mutates what it was handed
# ---------------------------------------------------------------------------

def test_the_history_is_not_modified():
    # It is `req.tokens` — the live request. Slicing is enough to keep this
    # true, but the test pins it: an in-place optimisation here would corrupt
    # the sequence rather than merely propose badly.
    tokens = [1, 2, 3, 4, 1, 2]
    before = list(tokens)
    lookup(tokens)
    assert tokens == before
