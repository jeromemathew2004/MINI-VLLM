"""Phase 6 — the speculative-decoding regression guard.

Correctly implemented rejection sampling (Leviathan et al. 2023, Chen et al.
2023) returns the target model's exact distribution regardless of what the draft
proposes. At temperature 0 that means speculative decoding must reproduce plain
greedy decoding token for token. That property, not throughput, is what these
tests protect: a speedup from a subtly wrong verify pass is worthless, and the
failure is silent — the model keeps emitting fluent text, just not the text it
should.

Three levels, cheapest first:

1. **The maths.** `Sampler.rejection_sample` against synthetic distributions. No
   models, no GPU, no checkpoints — runs anywhere, in under a second.
2. **The round.** One `verify()` + rejection-sample round must emit exactly what
   K+1 plain decode steps would, with acceptance scripted across every M from 0
   to K rather than left to a draft model. Needs the GPU and the target
   checkpoint; marked `gpu`.
3. **The engine.** Greedy output identical with speculation on and off, through
   `Engine.generate`. Minutes and two model loads, so it lives in
   `test_spec_decode_end_to_end.py` behind `--slow`.

**On exactness.** Level 2 cannot assert bitwise equality against plain decoding
and it is important to know why before "fixing" a failure here. The model's
logits depend on the batch shape it runs at — decoding one request at batch 1
and at batch 5 differs by up to 0.5 absolute on Qwen3-0.6B in bf16, with no
speculative decoding involved at all, because cuBLAS retiles per shape. A verify
pass has K+1 query rows where a decode step has 1, so its logits are
*necessarily* not bitwise equal. `batch_shape_noise_floor` measures that floor
and a divergence is tolerated only where the target's own top-2 gap sits below
it, i.e. at a genuine tie a sub-ULP perturbation may reorder. Everything else is
a hard failure.

**If these fail, check the backend first.** `w4096/mini-flash-attention` ships a
shared-memory race in its decode kernel that makes attention nondeterministic at
batch width >= 6, and a verify pass runs at width `num_requests * (K+1)`.
`patches/mini-flash-attention-decode-race.patch` fixes it and must be reapplied
to any rebuild. `python experiments/decode_determinism_check.py` settles that
question in one run, and misreading a missing patch as a rejection-sampling bug
is the exact trap the phase ordering exists to avoid.

Usage:
    pytest tests/                          # maths only, if there is no GPU
    pytest tests/ -v                       # maths + round, on this host
    pytest tests/ --slow                   # everything
"""

import pytest
import torch

from minivllm.models.layers.sampler import Sampler
from tests import spec_harness as harness

K = 4


# ===========================================================================
# level 1 — the maths. No models, no GPU, no checkpoints.
# ===========================================================================
#
# Every tensor below is created with an explicit device="cpu". That is not
# noise: `Executor.__init__` calls torch.set_default_device("cuda") as a global
# side effect, so once any GPU test in this session has built an engine, an
# unqualified torch.rand lands on the GPU and stops matching a CPU generator.

def _mismatched_distributions(num_draft: int, vocab: int, generator: torch.Generator):
    """A p and q that disagree hard, with a token q loves and p does not.

    The draft will keep proposing token 0 and it must keep being rejected — the
    regime where a broken sampler leaks the draft's preferences into the output.
    """
    p = torch.rand(num_draft + 1, vocab, generator=generator, device="cpu") + 0.05
    q = torch.rand(num_draft, vocab, generator=generator, device="cpu") + 0.05
    q[:, 0] += 3.0
    p[:, 0] = 0.01
    p /= p.sum(-1, keepdim=True)
    q /= q.sum(-1, keepdim=True)
    return p, q


@pytest.mark.parametrize("num_draft", [1, K])
def test_rejection_sampling_returns_the_target_distribution(num_draft):
    """The first token out of a round is distributed exactly as p, whatever q is.

    This is the Leviathan/Chen lemma, checked directly on the function the
    engine calls. Every trial is one row of a single batched call, which also
    makes it the only direct test that batching preserved the per-request
    independence the maths assumes.
    """
    trials = 20000
    vocab = 8
    generator = torch.Generator(device="cpu").manual_seed(0)
    p, q = _mismatched_distributions(num_draft, vocab, generator)

    draft_tokens = torch.multinomial(q, trials, replacement=True,
                                     generator=generator).T.contiguous()
    results = Sampler().rejection_sample(
        draft_tokens,
        q.unsqueeze(0).expand(trials, -1, -1),
        p.unsqueeze(0).expand(trials, -1, -1),
        generator,
    )

    counts = torch.zeros(vocab, device="cpu")
    for tokens, _ in results:
        counts[tokens[0]] += 1

    empirical = counts / trials
    stderr = (p[0] * (1 - p[0]) / trials).sqrt()
    worst = float(((empirical - p[0]).abs() / stderr).max())

    assert worst < 5.0, (
        f"output distribution does not match the target: {worst:.2f} sigma\n"
        f"  expected  {p[0].tolist()}\n"
        f"  empirical {empirical.tolist()}"
    )


@pytest.mark.parametrize("num_draft", [1, 2, K])
def test_a_round_always_makes_forward_progress(num_draft):
    """`len(tokens) == num_accepted + 1`, so a round can never emit nothing.

    This is what guarantees the engine still terminates when the draft is
    useless — the residual resample on rejection, or the bonus token on full
    acceptance, is always there.
    """
    trials = 500
    vocab = 8
    generator = torch.Generator(device="cpu").manual_seed(1)
    p, q = _mismatched_distributions(num_draft, vocab, generator)

    draft_tokens = torch.multinomial(q, trials, replacement=True,
                                     generator=generator).T.contiguous()
    results = Sampler().rejection_sample(
        draft_tokens,
        q.unsqueeze(0).expand(trials, -1, -1),
        p.unsqueeze(0).expand(trials, -1, -1),
        generator,
    )

    assert len(results) == trials
    proposed = draft_tokens.tolist()
    for row, (tokens, num_accepted) in enumerate(results):
        assert len(tokens) == num_accepted + 1
        assert 0 <= num_accepted <= num_draft
        # The accepted prefix is the draft's own tokens, unmodified — a round
        # may only ever truncate a proposal, never rewrite one it kept.
        assert tokens[:num_accepted] == proposed[row][:num_accepted]


def test_greedy_degenerates_to_agree_or_take_the_target():
    """With one-hot p and q, a round is 'accept while the draft agrees'.

    Greedy is not a special case in the sampler — it is a one-hot distribution
    fed through the general path (`Sampler.greedy_probs`). That is what makes
    the greedy exact-match tests a test of the general path rather than of a
    shortcut, so it is worth pinning the degenerate behaviour explicitly.
    """
    vocab = 16
    target = [3, 9, 9, 1]
    proposed = [3, 9, 4, 1]  # diverges at index 2
    num_draft = len(target)

    p_logits = torch.zeros(1, num_draft + 1, vocab, device="cpu")
    for i, token in enumerate(target):
        p_logits[0, i, token] = 1.0
    p_logits[0, num_draft, 7] = 1.0  # the bonus token, if everything is accepted

    q_logits = torch.zeros(1, num_draft, vocab, device="cpu")
    for i, token in enumerate(proposed):
        q_logits[0, i, token] = 1.0

    sampler = Sampler()
    (tokens, num_accepted), = sampler.rejection_sample(
        torch.tensor([proposed], device="cpu"),
        sampler.greedy_probs(q_logits),
        sampler.greedy_probs(p_logits),
    )

    # Accepts 3 and 9, rejects 4, and resamples the residual — which is one-hot
    # on the target's own choice, so the round emits exactly greedy's tokens.
    assert num_accepted == 2
    assert tokens == [3, 9, 9]


def test_full_acceptance_takes_the_bonus_token():
    vocab = 16
    agreed = [3, 9, 2, 1]

    p_logits = torch.zeros(1, len(agreed) + 1, vocab, device="cpu")
    for i, token in enumerate(agreed):
        p_logits[0, i, token] = 1.0
    p_logits[0, len(agreed), 7] = 1.0

    q_logits = torch.zeros(1, len(agreed), vocab, device="cpu")
    for i, token in enumerate(agreed):
        q_logits[0, i, token] = 1.0

    sampler = Sampler()
    (tokens, num_accepted), = sampler.rejection_sample(
        torch.tensor([agreed], device="cpu"),
        sampler.greedy_probs(q_logits),
        sampler.greedy_probs(p_logits),
    )

    assert num_accepted == len(agreed)
    assert tokens == agreed + [7]


# ===========================================================================
# level 2 — one round against the real target model
# ===========================================================================

pytestmark_reason = harness.missing_requirements()
requires_gpu = pytest.mark.skipif(bool(pytestmark_reason), reason=pytestmark_reason or "")


@pytest.fixture(scope="module")
def engine():
    """One engine for every GPU test in this module.

    Module-scoped rather than session-scoped so its memory is released before
    `test_spec_decode_end_to_end.py` starts building its own — on a 4 GB card a
    second engine cannot be built while this one is alive.
    """
    eng = harness.build_engine()
    yield eng
    del eng
    harness.free_gpu_memory()


@pytest.fixture(scope="module")
def prompt(engine):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(harness.TARGET)
    return harness.chat_prompts(tokenizer, harness.PROMPTS[:1])[0]


@pytest.fixture(scope="module")
def noise_floor(engine, prompt):
    return harness.batch_shape_noise_floor(engine, prompt, K + 1)


@pytest.fixture(scope="module")
def anchors(engine, prompt):
    """A greedy reference walk, plus the offsets along it to test at.

    Anchors are spread deliberately: a very short context, a mid one, and one
    past the 64-token block boundary so a second KV-cache page is in play.
    """
    eos_ids = set(engine.config.eos_token_ids or set())
    reference = harness.reference_walk(engine, prompt, 64, eos_ids)
    return reference, [g for g in (1, 24, 60) if g < len(reference)]


@requires_gpu
@pytest.mark.gpu
def test_verify_with_no_proposals_matches_decode_bitwise(engine, prompt):
    """`verify()` with K=0 is a single query row — the same shape as decode.

    The batch-shape argument does not apply here, so this must agree *exactly*.
    It is the sharpest assertion available anywhere in this feature, and it pins
    down `positions`, `cache_seqlens`, `slot_mapping` and `block_table` with no
    slack at all.
    """
    req, _ = harness.prefill_one(engine, prompt)
    expected = harness.decode_logits(engine, [req])[0]
    got = engine.executor.verify([req], [[]])[0, 0].float()
    harness.release(engine, req)

    assert float((expected - got).abs().max()) == 0.0


@requires_gpu
@pytest.mark.gpu
def test_ngram_proposer_matches_the_draft_models_contract(engine):
    """`propose_ngram` must be indistinguishable from a greedy draft downstream.

    That is the entire reason the n-gram proposer needed no changes to
    `verify()`, `rejection_sample()` or the scheduler: a lookup is deterministic,
    so its `q` is one-hot on the token it proposed, which is exactly what
    `Sampler.greedy_probs` returns for a draft model. If that stopped holding —
    say `q` were left as counts, or built off a stale proposal list — rejection
    sampling would silently compare the wrong probabilities and the output would
    drift from greedy without anything raising.

    Runs on the plain (non-speculative) engine: the proposer reads `req.tokens`
    and nothing else, so it needs no draft model and no second engine build.
    """
    from minivllm.engine.request import Request

    matching = Request([1, 2, 3, 4, 5, 1, 2, 3], harness.GREEDY)
    # Strictly increasing, so no token — let alone any pair — ever repeats.
    missing = Request(list(range(200, 260)), harness.GREEDY)

    proposals, q = engine.executor.propose_ngram([matching, missing])
    k = engine.config.num_speculative_tokens

    assert proposals[0] == [4, 5, 1, 2][:k]
    # A miss still contributes K tokens: verify() returns a dense
    # (batch, K+1, vocab) tensor and asserts uniform proposal length, so a
    # ragged batch would not survive it. Filler is rejected at i=0.
    assert len(proposals[1]) == k

    assert q.shape == (2, k, engine.config.hf_config.vocab_size)
    # device="cpu" spelled out: building the engine set torch's default device
    # to cuda as a global side effect (see the note at the top of this file).
    assert torch.equal(q.argmax(-1).cpu(), torch.tensor(proposals, device="cpu"))
    assert torch.equal(q.sum(-1).cpu(), torch.ones(2, k, device="cpu"))


@requires_gpu
@pytest.mark.gpu
def test_ngram_proposer_declines_when_nothing_matches(engine):
    """No match anywhere in the batch means no round at all.

    Not an optimisation: a round costs ~1.4x a graphed decode step, so
    speculating on filler that cannot be accepted is a straight loss. The
    `(None, None)` return is what `execute_speculative` falls back on, and it is
    the only thing keeping the lookup proposer from being a pessimisation on
    open-ended text.
    """
    from minivllm.engine.request import Request

    proposals, q = engine.executor.propose_ngram(
        [Request(list(range(200, 260)), harness.GREEDY)])

    assert proposals is None
    assert q is None


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("num_correct", range(K + 1))
def test_round_emits_what_greedy_decoding_would(engine, prompt, anchors, noise_floor,
                                                num_correct):
    """A round with exactly `num_correct` correct proposals emits `truth[:M+1]`.

    Parametrised over M so a failure names its own acceptance regime. M=0 is the
    case implementations get wrong first: every proposal is rejected, the whole
    proposal region of the KV cache goes stale, and the round must still emit
    the one token the target wanted. M=K is the other end — full acceptance,
    where the emitted token comes from the bonus row instead of the residual.
    """
    reference, offsets = anchors
    vocab_size = engine.config.hf_config.vocab_size

    for g in offsets:
        prefix = prompt + reference[:g]

        # Ground truth: what K+1 plain decode steps produce from a fresh prefill
        # of this exact prefix. Freshly anchored rather than free-running,
        # because these passes *write* the KV cache as well as read it, so a
        # decode walk and a verify walk lay down subtly different K/V and drift
        # apart over dozens of rounds. That drift says nothing about
        # correctness; an earlier version of this check measured it by mistake.
        truth, _, gaps = harness.sequential_steps(engine, prefix, K + 1)
        proposal = harness.scripted_proposal(truth, num_correct, K, vocab_size)

        req, _ = harness.prefill_one(engine, prefix)
        tokens, num_accepted = harness.run_round(engine, req, proposal)
        harness.release(engine, req)

        assert len(tokens) == num_accepted + 1

        expected = truth[:num_correct + 1]
        if num_accepted == num_correct and tokens == expected:
            continue

        # Only forgivable where the target itself was within noise of a tie at
        # the step that went wrong.
        first_bad = next((j for j in range(min(len(tokens), len(expected)))
                          if tokens[j] != expected[j]),
                         min(num_accepted, num_correct))
        gap = gaps[min(first_bad, len(gaps) - 1)]
        assert gap <= noise_floor, (
            f"round at anchor g={g} with M={num_correct} accepted {num_accepted} "
            f"and emitted {tokens}, wanted {num_accepted == num_correct and 'M' or num_correct} "
            f"accepted and {expected}. The top-2 logit gap at step {first_bad} is "
            f"{gap:.3f}, above the {noise_floor:.3f} batch-shape noise floor, so this "
            f"is not a tie being reordered."
        )
