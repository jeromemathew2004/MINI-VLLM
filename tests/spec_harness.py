"""Primitives for driving the executor by hand, shared by tests and experiments.

Speculative decoding is a decode-path feature, so most of what has to be checked
about it sits below `Engine.generate`: a request has to be prefilled, left with
its cache in the exact state a round assumes, advanced by a known number of
plain decode steps, and then compared against a round that claims to do the
same thing in one pass. None of that is expressible through the public API, and
all of it is needed by both the pytest regression suite and the `experiments/`
gate scripts.

This module is the single definition. The dependency runs tests -> experiments
and not the other way around on purpose: `experiments/` is exploratory and gets
edited freely, and a regression suite that breaks when someone adjusts an
experiment is a regression suite nobody trusts.

Nothing here prints unless asked. The experiment scripts want prose; the tests
want silence.
"""

import gc
import os

import torch

from minivllm.config.config import Config
from minivllm.config.sampling import SamplingParams
from minivllm.engine.engine import Engine
from minivllm.engine.request import Request
from minivllm.scheduler.batch import Batch

TARGET = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-draft-random/")

PROMPTS = [
    "What is the meaning of life?",
    "How do I get started with LLMs?",
]

# Open-ended generation is the worst case for the n-gram proposer and the best
# case for exercising its skip path; this is the other end — prompts whose
# answer is largely a copy of the prompt, which is where prompt-lookup earns its
# keep. Kept here rather than in the test so the experiment scripts measure the
# same two workloads the regression suite does.
REPETITIVE_PROMPTS = [
    "Repeat the following sentence exactly five times, one per line, with no "
    "commentary: The quick brown fox jumps over the lazy dog.",
    "Echo this list back to me verbatim and in the same order, one item per "
    "line: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel.",
]

# Greedy: temperature 1.0 with top_k 0 and top_p 1.0 leaves all three tensors
# None in Sampler.forward, which takes the argmax branch and never imports
# flashinfer (no reliable Windows wheel). It is also the only sampling policy
# Executor.execute_speculative accepts.
GREEDY = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, max_tokens=1 << 30)


def missing_requirements(draft: str | None = None) -> str:
    """Empty string when the GPU tests can run, else why they cannot."""
    if not torch.cuda.is_available():
        return "needs a CUDA device"
    if not os.path.isdir(TARGET):
        return f"needs the target checkpoint at {TARGET}"
    if draft is not None and not os.path.isdir(draft):
        return f"needs a draft checkpoint at {draft}"
    return ""


def build_config(spec: bool = False, draft: str = DRAFT, k: int = 4,
                 cuda_graph: bool = False, method: str = "draft",
                 ngram_min_match: int | None = None,
                 ngram_max_match: int | None = None) -> Config:
    """The 4 GB RTX 3050 profile.

    The stock `kv_cache_block_size` of 256 costs 28 MiB per block for
    Qwen3-0.6B (2 * 28 layers * 256 * 8 kv heads * 128 head dim * 2 bytes),
    far too coarse when ~800 MiB of the 4 GB is already gone to the CUDA
    context and the desktop; 64 brings that to 7 MiB. The stock Config does
    not start on this card at all.

    CUDA graphs default off: they are a decode-path optimisation already shown
    equivalent to eager in Phase 3, and leaving them out keeps the number of
    variables under test down.

    The n-gram match bounds default to None rather than to a number, so the
    tests exercise whatever `Config` actually ships. Duplicating the default
    here would let the two drift and leave the shipped configuration untested.
    """
    ngram_bounds = {}
    if ngram_min_match is not None:
        ngram_bounds["ngram_min_match_len"] = ngram_min_match
    if ngram_max_match is not None:
        ngram_bounds["ngram_max_match_len"] = ngram_max_match

    return Config(
        model=TARGET,
        max_model_len=1024,
        max_num_batched_tokens=2048,
        max_num_batched_seqs=8,
        kv_cache_block_size=64,
        gpu_memory_utilization=0.9,
        use_cuda_graph=cuda_graph,
        use_speculative_decoding=spec,
        speculative_method=method,
        draft_model=draft if spec and method == "draft" else "",
        num_speculative_tokens=k,
        **ngram_bounds,
    )


def build_engine(spec: bool = False, draft: str = DRAFT, k: int = 4,
                 cuda_graph: bool = False, method: str = "draft",
                 ngram_min_match: int | None = None,
                 ngram_max_match: int | None = None) -> Engine:
    # Collect first, unconditionally. A previously-built engine may still be
    # holding its weights and cache through a reference cycle, and on a 4 GB
    # card that is the difference between sizing a KV cache and failing to.
    free_gpu_memory()
    return Engine(build_config(spec, draft, k, cuda_graph, method,
                               ngram_min_match, ngram_max_match))


def chat_prompts(tokenizer, prompts: list[str] = PROMPTS,
                 enable_thinking: bool = True) -> list[list[int]]:
    """Tokenised chat prompts.

    `enable_thinking=False` for the repetitive set: Qwen3 spends its first few
    dozen tokens reasoning, and a short generation budget would be consumed
    entirely by the think block, leaving no copied text for a lookup proposer to
    find and nothing for the workload to actually measure.
    """
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking)
        for p in prompts
    ]


def free_gpu_memory() -> None:
    """Collect and release cached GPU memory. Call after dropping an Engine.

    nn.Module graphs contain reference cycles, so refcounting alone will not
    free the weights or the KV cache when an engine goes out of scope. Without
    the collect, a second engine built afterwards sees the first one's memory
    still resident and sizes its cache down — or trips the
    `kv_cache_num_blocks > 0` assert outright on a 4 GB card.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# driving the executor by hand
# ---------------------------------------------------------------------------

def prefill_one(engine: Engine, prompt_tokens: list[int]) -> tuple[Request, int]:
    """Submit one request and run its prefill, returning it and its first token.

    Afterwards `req.tokens` is `prompt + [first]` and the cache holds the
    prompt — the last committed token has not been fed through the model, which
    is the invariant every decode step and every verify round assumes.
    """
    req = Request(prompt_tokens, GREEDY)
    engine.scheduler.submit(req)
    batch = engine.scheduler.schedule()
    assert batch is not None and batch.type == Batch.PREFILL and batch.requests == [req]
    tokens = engine.executor.execute(batch)
    # Bypass Scheduler.update: it also runs prefix-cache bookkeeping and
    # end-of-sequence handling, neither of which this harness wants.
    req.append_output_token(tokens[0])
    return req, tokens[0]


def release(engine: Engine, req: Request) -> None:
    if req in engine.scheduler.running:
        engine.scheduler.running.remove(req)
    if req.blocks:
        engine.scheduler.block_manager.deallocate(req)


def decode_logits(engine: Engine, requests: list[Request]) -> torch.Tensor:
    """One plain decode step for `requests`, eager, returning (batch, vocab)."""
    for req in requests:
        engine.scheduler.block_manager.allocate_block_for_decode(req)
    with torch.inference_mode():
        input_ids, ctx = engine.executor._build_decode_input(requests)
        return engine.executor.model(ctx, input_ids, ctx.positions).float()


def reference_walk(engine: Engine, prompt_tokens: list[int], max_tokens: int,
                   eos_ids: set[int]) -> list[int]:
    """Plain greedy decoding — the token sequence anchors are taken from."""
    req, first = prefill_one(engine, prompt_tokens)
    tokens = [first]

    while len(tokens) < max_tokens and tokens[-1] not in eos_ids:
        nxt = int(decode_logits(engine, [req])[0].argmax())
        tokens.append(nxt)
        req.append_output_token(nxt)

    release(engine, req)
    return tokens


def sequential_steps(engine: Engine, prefix: list[int], steps: int
                     ) -> tuple[list[int], list[torch.Tensor], list[float]]:
    """Prefill `prefix`, then take `steps` plain decode steps.

    Returns the tokens produced (the first from prefill, the rest from decode),
    their logit rows, and each step's top-2 gap. This is the behaviour a verify
    pass, and a whole speculative round, has to reproduce.
    """
    req, first = prefill_one(engine, prefix)
    with torch.inference_mode():
        tokens = [first]
        rows: list[torch.Tensor] = []
        gaps: list[float] = []

        for _ in range(steps):
            logits = decode_logits(engine, [req])[0]
            top2 = torch.topk(logits, 2).values
            rows.append(logits.clone())
            gaps.append(float(top2[0] - top2[1]))
            nxt = int(logits.argmax())
            tokens.append(nxt)
            req.append_output_token(nxt)

    release(engine, req)
    # `tokens[0]` came from the prefill; rows[i] is the step that produced
    # tokens[i + 1].
    return tokens[1:], rows, gaps


def run_round(engine: Engine, req: Request, proposal: list[int]) -> tuple[list[int], int]:
    """One verify + rejection-sample round on scripted proposals.

    Bypasses `Executor.propose` on purpose. `q` is built as a greedy draft's
    would be — one-hot on the token proposed — so the only thing under test is
    the target's side of the round. Returns `(tokens, num_accepted)`.
    """
    sampler = engine.executor.sampler
    engine.scheduler.block_manager.allocate_block_for_decode(req, extra_tokens=len(proposal))

    target_logits = engine.executor.verify([req], [proposal])
    p = sampler.greedy_probs(target_logits)

    draft_tokens = torch.tensor([proposal], dtype=torch.int64, device=p.device)
    q = torch.zeros(1, len(proposal), p.shape[-1], device=p.device)
    q.scatter_(-1, draft_tokens.unsqueeze(-1), 1.0)

    (tokens, num_accepted), = sampler.rejection_sample(draft_tokens, q, p)
    return tokens, num_accepted


def scripted_proposal(truth: list[int], num_correct: int, k: int, vocab_size: int) -> list[int]:
    """K proposals of which exactly the first `num_correct` will be accepted.

    `truth[i]` is the token that follows query row i of a verify pass, so a
    proposal is accepted precisely when it equals it.
    """
    return [t if i < num_correct else (t + 1) % vocab_size
            for i, t in enumerate(truth[:k])]


# ---------------------------------------------------------------------------
# the control: how much does batch shape alone move the logits?
# ---------------------------------------------------------------------------

def batch_shape_noise_floor(engine: Engine, prompt_tokens: list[int], width: int,
                            verbose: bool = False) -> float:
    """Decode one state at batch size 1 and at batch size `width`.

    No verify pass anywhere. Whatever this returns is the floor below which
    "different logits" means "different GEMM tiling", not "different maths":
    the model's output depends on the batch shape it runs at, because cuBLAS
    picks different tilings per shape. A verify pass changes the batch shape by
    construction (K+1 rows where decode has 1), so any claim about it has to be
    stated against this number rather than against zero.
    """
    requests = [prefill_one(engine, prompt_tokens)[0] for _ in range(width)]

    single = decode_logits(engine, requests[:1])[0]
    batched = decode_logits(engine, requests)

    floor = max(float((single - batched[i]).abs().max()) for i in range(width))
    spread = max(float((batched[0] - batched[i]).abs().max()) for i in range(1, width))
    mean_dev = float((single - batched[0]).abs().mean())

    for req in requests:
        release(engine, req)

    if verbose:
        print(f"\n== control: plain decode at batch 1 vs batch {width} (no verify pass) ==")
        print(f"   max|batch1 - batch{width}| = {floor:.4f}   mean = {mean_dev:.4f}")
        print(f"   spread among identical rows within one batch = {spread:.4f} "
              f"(deterministic within a shape)")
        print(f"   => noise floor for this model/dtype: {floor:.4f}")

    return floor
