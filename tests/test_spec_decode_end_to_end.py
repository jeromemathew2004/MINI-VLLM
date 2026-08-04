"""Phase 6, level 3 — greedy output identical with speculation on and off.

This is the strictest test in the project and the slowest: it runs the real
`Engine.generate` with the real scheduler, the real paged cache and the real
draft model, twice, and compares token ids. Everything the unit tests stub out
is live here — block allocation and preemption, multi-token commits, EOS
truncation mid-round, the draft's own KV cache.

It is behind `--slow` because it loads model checkpoints twice per case and
takes minutes. Each case builds and frees its own engines rather than sharing a
fixture: on a 4 GB card two engines cannot coexist, which is also why this lives
in its own module — a module-scoped fixture in
`test_spec_decode_correctness.py` is torn down before this file is collected.

**Two drafts, because they exercise different code.** A random-init draft is
rejected at i=0 in every round, so it only ever drives the residual-resample
path and leaves the whole proposal region stale. The target drafting for itself
accepts every proposal, which is the *only* regime where the draft ends a round
behind the committed sequence and the catch-up row in `Executor.propose` does
real work — if that row were wrong the draft would propose from a stale cache
and acceptance would collapse after round one.

**On "identical".** Byte-identical greedy output holds in every configuration
measured so far, but it is a tolerance rather than a theorem: batch shape moves
logits by up to 0.5 absolute (cuBLAS retiling per shape, nothing to do with
speculation), and a near-tie can flip an emitted token without anything being
wrong. A failure here is a real signal worth investigating, not something to
relax on sight — but investigate it by checking whether the divergent step was a
near-tie before concluding the round is broken.

Usage:
    pytest tests/test_spec_decode_end_to_end.py --slow -v
"""

import pytest

from tests import spec_harness as harness

K = 4
MAX_TOKENS = 48

_missing = harness.missing_requirements(harness.DRAFT)
requires_gpu = pytest.mark.skipif(bool(_missing), reason=_missing or "")


class RoundRecorder:
    """Wraps `Executor.execute_speculative` to check every round's invariants."""

    def __init__(self, engine, k: int):
        self.k = k
        self.rounds = 0
        self.emitted = 0
        self.accepted = 0
        self.skipped = 0
        self.histogram = [0] * (k + 1)
        self.violations: list[str] = []

        self.engine = engine
        self.block_size = engine.config.kv_cache_block_size
        self._inner = engine.executor.execute_speculative
        engine.executor.execute_speculative = self._record

    def detach(self):
        """Drop every reference to the engine, keeping only the counters.

        `self._inner` is a *bound* method, so holding it pins the executor, the
        model weights and the KV cache. Without this the engine is never
        collected and the next one has no memory left to size a cache against —
        a leak whose symptom ("not enough VRAM") points nowhere near its cause.
        """
        self.engine.executor.execute_speculative = self._inner
        self._inner = None
        self.engine = None

    def _record(self, batch):
        before = [len(req.tokens) for req in batch.requests]
        tokens, num_accepted, num_proposed = self._inner(batch)
        if num_proposed == 0:
            # The proposer declined and the step fell back to plain decode.
            # Only the n-gram proposer does this; a draft model always proposes.
            self.skipped += 1

        for req, length, emitted, accepted in zip(batch.requests, before, tokens, num_accepted):
            self.rounds += 1
            self.emitted += len(emitted)
            self.accepted += accepted
            self.histogram[accepted] += 1

            if len(emitted) != accepted + 1:
                self.violations.append(
                    f"request {req.id}: emitted {len(emitted)} tokens for {accepted} accepted")
            if not 0 <= accepted <= self.k:
                self.violations.append(f"request {req.id}: accepted {accepted} of {self.k}")
            # Committing is the scheduler's job. If the executor also did it,
            # tokens would be silently duplicated.
            if len(req.tokens) != length:
                self.violations.append(
                    f"request {req.id}: execute_speculative mutated req.tokens")
            # Every position the round wrote KV for must be backed by a block.
            needed = -(-(length + self.k) // self.block_size)
            if len(req.blocks) < needed:
                self.violations.append(
                    f"request {req.id}: {len(req.blocks)} blocks, needs {needed} for "
                    f"{length} tokens + {self.k} proposals")

        return tokens, num_accepted, num_proposed


def _generate(spec: bool, draft: str, k: int, prompts, cuda_graph: bool = False,
              method: str = "draft", max_tokens: int = MAX_TOKENS):
    engine = harness.build_engine(spec=spec, draft=draft, k=k, cuda_graph=cuda_graph,
                                  method=method)
    if cuda_graph:
        # Guard against the test quietly not exercising what it claims to: if
        # graph capture were skipped, every assertion below would still pass
        # while measuring the eager path.
        assert engine.executor.graph_runner is not None
        if spec and method == "draft":
            assert engine.executor.draft_graph_runner is not None
    if spec and method == "ngram":
        # The lookup proposer runs no model, so there must be no draft loaded
        # and no draft cache charged against the memory budget. Pinning it here
        # because "it still works" would not notice a draft being loaded and
        # ignored — it would just quietly cost ~50 KV-cache blocks.
        assert engine.executor.draft_model is None
        assert engine.executor.draft_kv_cache is None
    recorder = RoundRecorder(engine, k) if spec else None

    sampling = harness.SamplingParams(temperature=1.0, top_k=0, top_p=1.0,
                                      max_tokens=max_tokens)
    outputs = engine.generate(prompts, sampling, use_tqdm=False)
    stats = engine.metrics.stats()
    tokens = [o["tokens"] for o in outputs]

    if recorder is not None:
        recorder.detach()
    del engine
    harness.free_gpu_memory()
    return tokens, recorder, stats


REPETITIVE_MAX_TOKENS = 96


@pytest.fixture(scope="module")
def prompts():
    from transformers import AutoTokenizer
    return harness.chat_prompts(AutoTokenizer.from_pretrained(harness.TARGET))


@pytest.fixture(scope="module")
def repetitive_prompts():
    """Prompts whose answer is largely a copy of the question.

    The n-gram proposer only has something to propose when the text repeats, so
    testing it on open-ended prompts would exercise its skip path and nothing
    else. Thinking is disabled: Qwen3 spends its first few dozen tokens
    reasoning, which would eat the whole generation budget before any copying
    starts.
    """
    from transformers import AutoTokenizer
    return harness.chat_prompts(AutoTokenizer.from_pretrained(harness.TARGET),
                                harness.REPETITIVE_PROMPTS, enable_thinking=False)


@pytest.fixture(scope="module")
def baseline(prompts):
    """Plain greedy decoding, speculation off. The thing to be reproduced.

    Returns a memoising lookup rather than a single result, because a run with
    CUDA graphs has to be compared against a *graphed* baseline. Comparing a
    graphed speculative run against an eager baseline would fold graph-vs-eager
    numerics into the diff and blame speculation for them.
    """
    cache: dict[bool, list[list[int]]] = {}

    def get(cuda_graph: bool):
        if cuda_graph not in cache:
            cache[cuda_graph], _, _ = _generate(False, harness.DRAFT, K, prompts,
                                                cuda_graph=cuda_graph)
        return cache[cuda_graph]

    return get


@requires_gpu
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_random_draft_reproduces_greedy(prompts, baseline, cuda_graph):
    """Zero acceptance: every round rejects at i=0 and emits one token.

    The draft is useless by construction, so this is the case where a round has
    to fall all the way back to the target's own choice — and where forward
    progress depends entirely on the residual resample always being there.

    Run with graphs off *and* on. Graphs are not a detail here: a speculative
    round replays them at width `B*(K+1)` rather than `B`, which is a shape the
    non-speculative engine never produces, and without them speculation cannot
    break even at any acceptance rate (`experiments/spec_breakeven.py`).
    """
    tokens, recorder, stats = _generate(True, harness.DRAFT, K, prompts,
                                        cuda_graph=cuda_graph)

    assert recorder.violations == []
    assert recorder.rounds > 0
    assert stats.acceptance_rate == 0.0, (
        "a randomly-initialised draft sharing a 151936-token vocabulary should "
        f"essentially never be accepted, got {stats.acceptance_rate:.1%}"
    )
    assert recorder.emitted == recorder.rounds, "zero acceptance must emit one token per round"
    assert tokens == baseline(cuda_graph)


@requires_gpu
@pytest.mark.slow
@pytest.mark.gpu
def test_self_draft_reproduces_greedy(prompts, baseline):
    """Full acceptance: the target drafts for itself, so every proposal lands.

    This is the only end-to-end case that exercises `Executor.propose`'s
    catch-up row, since it is the only one where a round commits K+1 tokens and
    leaves the draft a token behind. Needs VRAM for two copies of the target and
    skips, rather than fails, when there is not enough.
    """
    try:
        tokens, recorder, stats = _generate(True, harness.TARGET, K, prompts)
    except AssertionError as exc:
        if "No memory left for the KV cache" not in str(exc):
            raise
        pytest.skip(f"not enough VRAM for two copies of the target: {exc}")

    assert recorder.violations == []
    assert recorder.rounds > 0
    # A model drafting for itself agrees with itself, save for the odd near-tie
    # where the draft's batch width differs from the verify pass's. Rejection
    # sampling absorbs those without changing the output, which is the point.
    assert stats.acceptance_rate > 0.95, (
        f"the target drafting for itself accepted only {stats.acceptance_rate:.1%}; "
        f"below ~95% suggests the draft is proposing from a stale cache"
    )
    assert stats.tokens_per_request_step > K, (
        f"{stats.tokens_per_request_step:.2f} tokens per request per step at "
        f"{stats.acceptance_rate:.1%} acceptance — speculation is not compounding"
    )
    assert tokens == baseline(False)


@requires_gpu
@pytest.mark.slow
@pytest.mark.gpu
def test_ngram_proposer_reproduces_greedy(repetitive_prompts):
    """The lookup proposer, on the workload it is actually for.

    No draft model is involved at all — proposals come from the request's own
    token history. The correctness claim is unchanged and that is the point:
    rejection sampling returns the target's distribution whatever the proposer
    offers, so swapping the proposer may move the acceptance rate but must never
    move a token.

    Also asserts the proposer is *doing something*. Byte-identical output is
    trivially satisfiable by never speculating, and since the round skips
    whenever nothing matches, a proposer that silently matched nothing would
    pass the identity check while being a no-op.
    """
    baseline, _, _ = _generate(False, harness.DRAFT, K, repetitive_prompts,
                               max_tokens=REPETITIVE_MAX_TOKENS)
    tokens, recorder, stats = _generate(True, None, K, repetitive_prompts,
                                        method="ngram",
                                        max_tokens=REPETITIVE_MAX_TOKENS)

    assert recorder.violations == []
    assert recorder.rounds > 0
    assert stats.speculation_rate > 0.0, (
        "every decode step declined to speculate, so this test asserted nothing "
        "about the proposer — check ngram_min_match_len against the workload"
    )
    assert stats.acceptance_rate > 0.0, (
        f"the lookup proposer was accepted {stats.acceptance_rate:.1%} of the time on "
        f"text that copies its own prompt; at zero it is proposing but never landing, "
        f"which is a bug in the proposal offsets rather than a bad workload"
    )
    assert tokens == baseline


@requires_gpu
@pytest.mark.slow
@pytest.mark.gpu
def test_ngram_proposer_reproduces_greedy_on_open_ended_text(prompts, baseline):
    """The case the lookup proposer is *not* for, which still has to be correct.

    Open-ended generation is where the tail rarely appears earlier, so this run
    is dominated by the skip path — rounds that fall back to a plain decode step.
    That fallback is a second code path through `execute_speculative`, and it
    would be easy for it to double-commit or drop a token without any of the
    round-shaped assertions noticing.
    """
    tokens, recorder, _ = _generate(True, None, K, prompts, method="ngram")

    assert recorder.violations == []
    assert recorder.rounds > 0
    assert tokens == baseline(False)
