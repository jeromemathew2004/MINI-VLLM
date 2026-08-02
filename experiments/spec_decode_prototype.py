"""Phase 2 — standalone speculative-decoding prototype.

Deliberately outside `minivllm/`: this script uses plain HuggingFace models and
`DynamicCache`, no paging, no scheduler, no mini-flash-attention. Its only job
is to prove the rejection-sampling maths is correct *before* any of it is wired
into the engine (Phases 3-5).

Two gates are checked, in increasing order of strength:

1. `test_rejection_sampling_distribution` — the sampler is exercised against
   synthetic p/q distributions and the empirical output distribution is
   compared against p. This is the core lemma of Leviathan et al. / Chen et
   al.: the token returned by one draft-then-verify step is distributed
   exactly as p, whatever q is.
2. `test_greedy_exact_match` — with the target at temperature 0, speculative
   decoding must reproduce plain greedy decoding token for token, for several
   prompts and several values of K.

Usage:
    python experiments/spec_decode_prototype.py
    python experiments/spec_decode_prototype.py --model ~/huggingface/Qwen3-0.6B/ --device cuda
    python experiments/spec_decode_prototype.py --skip-model   # maths gate only
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM

DEFAULT_MODEL = "~/huggingface/Qwen3-0.6B/"

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley.",
]


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------

def probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Rows of `logits` -> rows of probabilities.

    Temperature 0 is represented as a one-hot distribution on the argmax rather
    than as a special case in the sampler, so that greedy decoding runs through
    exactly the same rejection-sampling code path as temperature > 0. That is
    what makes the greedy exact-match test a real test of the general path.
    """
    logits = logits.float()
    if temperature <= 0:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(logits / temperature, dim=-1)


def sample_from(probs: torch.Tensor, generator: torch.Generator) -> int:
    """Draw one token from a 1-D probability vector."""
    return int(torch.multinomial(probs, 1, generator=generator).item())


def make_generator(device, seed: int) -> torch.Generator:
    """RNG on `device` — torch requires the generator and the tensors to agree."""
    return torch.Generator(device=device).manual_seed(seed)


# ---------------------------------------------------------------------------
# rejection sampling — the part that has to be exactly right
# ---------------------------------------------------------------------------

def rejection_sample(
    draft_tokens: list[int],
    q: torch.Tensor,
    p: torch.Tensor,
    generator: torch.Generator,
) -> tuple[list[int], int]:
    """Modified rejection sampling (Leviathan et al. 2023, Chen et al. 2023).

    Args:
        draft_tokens: the K tokens the draft proposed.
        q: (K, V) — the draft's distribution at each proposal step. Row i is the
           distribution `draft_tokens[i]` was actually drawn from.
        p: (K+1, V) — the target's distribution at the same K positions, plus a
           (K+1)-th row for the bonus token that follows a full acceptance.
        generator: RNG, passed explicitly so runs are reproducible.

    Returns:
        (tokens, num_accepted) where `tokens` is the accepted prefix of
        `draft_tokens` followed by exactly one extra token — either the
        residual resample (on rejection) or the bonus token (on full
        acceptance). So `len(tokens) == num_accepted + 1` always: a round can
        never return zero tokens, which is what guarantees forward progress
        even when the draft is useless.
    """
    num_draft = len(draft_tokens)
    assert q.shape[0] == num_draft
    assert p.shape[0] == num_draft + 1

    accepted: list[int] = []
    for i in range(num_draft):
        x = draft_tokens[i]
        p_x = float(p[i, x])
        q_x = float(q[i, x])

        # q_x == 0 is unreachable — x was drawn from q — but guard anyway so a
        # numerically-zeroed row can never produce a nan ratio.
        accept_prob = 1.0 if q_x <= 0.0 else min(1.0, p_x / q_x)
        if float(torch.rand((), generator=generator, device=p.device)) < accept_prob:
            accepted.append(x)
            continue

        # Rejected at i: resample from the renormalised residual max(0, p - q).
        residual = torch.clamp(p[i] - q[i], min=0.0)
        total = float(residual.sum())
        if total <= 0.0:
            # Only reachable through floating-point noise: a rejection implies
            # p != q somewhere, which implies a positive residual mass.
            residual = p[i]
            total = float(residual.sum())
        return accepted + [sample_from(residual / total, generator)], i

    # Every proposal accepted — the target's (K+1)-th distribution is free.
    return accepted + [sample_from(p[num_draft], generator)], num_draft


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_from(model, tokens: list[int], cache: DynamicCache, cached: int) -> torch.Tensor:
    """Feed `tokens[cached:]` into `model`, extending `cache`.

    Returns (num_new, V) logits — row j predicts the token after
    `tokens[cached + j]`.
    """
    assert cached < len(tokens), "nothing new to feed"
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokens[cached:]], dtype=torch.long, device=device)
    cache_position = torch.arange(cached, len(tokens), device=device)
    out = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )
    return out.logits[0].float()


@torch.inference_mode()
def greedy_generate(model, prompt_ids: list[int], max_new_tokens: int, eos_ids: set[int]) -> list[int]:
    """Plain one-token-at-a-time greedy decoding. The reference output."""
    tokens = list(prompt_ids)
    cache = DynamicCache()
    cached = 0
    generated: list[int] = []

    while len(generated) < max_new_tokens:
        logits = forward_from(model, tokens, cache, cached)
        cached = len(tokens)
        next_token = int(logits[-1].argmax())
        tokens.append(next_token)
        generated.append(next_token)
        if next_token in eos_ids:
            break

    return generated


class SpecStats:
    def __init__(self):
        self.rounds = 0
        self.proposed = 0
        self.accepted = 0
        self.emitted = 0
        self.seconds = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.emitted / self.rounds if self.rounds else 0.0

    def __str__(self) -> str:
        return (f"rounds={self.rounds} emitted={self.emitted} "
                f"acceptance={self.acceptance_rate:.1%} "
                f"tokens/round={self.tokens_per_round:.2f} "
                f"time={self.seconds:.1f}s")


@torch.inference_mode()
def speculative_generate(
    target,
    draft,
    prompt_ids: list[int],
    max_new_tokens: int,
    num_speculative_tokens: int,
    eos_ids: set[int],
    generator: torch.Generator,
    temperature: float = 0.0,
    draft_temperature: float | None = None,
    stats: SpecStats | None = None,
) -> list[int]:
    """Draft-and-verify decoding for a single sequence.

    Cache invariant, maintained at the top of every round: both caches hold KV
    for `tokens[:cached]` with `cached <= len(tokens) - 1`, i.e. the last token
    of the sequence is always still unprocessed. Rejected proposals are undone
    with `DynamicCache.crop`, which is the prototype's stand-in for the
    stale-KV problem Phase 4 has to solve inside the paged cache.
    """
    K = num_speculative_tokens
    if draft_temperature is None:
        draft_temperature = temperature
    stats = stats or SpecStats()
    started = time.perf_counter()

    tokens = list(prompt_ids)
    target_cache, draft_cache = DynamicCache(), DynamicCache()
    target_cached = draft_cached = 0
    generated: list[int] = []

    while len(generated) < max_new_tokens:
        # --- 1. draft proposes K tokens autoregressively -------------------
        draft_tokens: list[int] = []
        q_rows: list[torch.Tensor] = []
        for _ in range(K):
            logits = forward_from(draft, tokens + draft_tokens, draft_cache, draft_cached)
            draft_cached = len(tokens) + len(draft_tokens)
            q = probs_from_logits(logits[-1], draft_temperature)
            q_rows.append(q)
            draft_tokens.append(sample_from(q, generator))

        # --- 2. target verifies all K in one pass --------------------------
        # Feeding tokens[target_cached:] covers the K proposals plus at least
        # the one already-sampled token that precedes them, so the last K+1
        # logit rows are exactly the distributions for proposals 0..K-1 plus
        # the bonus position.
        seq = tokens + draft_tokens
        logits = forward_from(target, seq, target_cache, target_cached)
        target_cached = len(seq)
        p = probs_from_logits(logits[-(K + 1):], temperature)

        # --- 3. rejection sampling ----------------------------------------
        new_tokens, num_accepted = rejection_sample(draft_tokens, torch.stack(q_rows), p, generator)

        stats.rounds += 1
        stats.proposed += K
        stats.accepted += num_accepted

        # --- 4. commit and roll the caches back ----------------------------
        # The verify pass wrote KV for all K proposals; everything past the
        # accepted prefix is stale and must go. The extra token in `new_tokens`
        # was never fed to either model, hence `- 1`.
        tokens = tokens + new_tokens
        target_cache.crop(min(target_cached, len(tokens) - 1))
        draft_cache.crop(min(draft_cached, len(tokens) - 1))
        target_cached = min(target_cached, len(tokens) - 1)
        draft_cached = min(draft_cached, len(tokens) - 1)

        for token in new_tokens:
            generated.append(token)
            if token in eos_ids or len(generated) == max_new_tokens:
                break
        if generated and (generated[-1] in eos_ids or len(generated) >= max_new_tokens):
            break

    stats.emitted += len(generated)
    stats.seconds += time.perf_counter() - started
    return generated


def build_random_draft(target_config, seed: int = 0):
    """A randomly-initialised tiny Qwen3 sharing the target's vocabulary.

    Phase 1 (training a real draft) is deferred on purpose: rejection sampling
    is distribution-preserving for *any* q, so draft quality moves the
    acceptance rate and nothing else. A random draft is therefore the harshest
    correctness case — near-zero acceptance means almost every round exercises
    the reject-at-i=0 path, which is the one implementations get wrong.
    """
    config = Qwen3Config(
        vocab_size=target_config.vocab_size,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=target_config.max_position_embeddings,
        rope_theta=target_config.rope_theta,
        rms_norm_eps=target_config.rms_norm_eps,
        tie_word_embeddings=True,
        dtype=torch.float32,
    )
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# gate 1 — the maths, no models involved
# ---------------------------------------------------------------------------

def test_rejection_sampling_distribution(trials: int = 40000, vocab: int = 8, seed: int = 0) -> None:
    """The first token out of a round must be distributed exactly as p.

    Runs the real `rejection_sample` (not a vectorised re-derivation of it) so
    the function under test is the one the engine will port.
    """
    print("\n== gate 1: rejection sampling reproduces the target distribution ==")
    generator = make_generator("cpu", seed)

    for K in (1, 4):
        # Deliberately mismatched p and q, with a token q likes and p does not.
        p = torch.rand(K + 1, vocab, generator=generator) + 0.05
        q = torch.rand(K, vocab, generator=generator) + 0.05
        q[:, 0] += 3.0
        p[:, 0] = 0.01
        p /= p.sum(-1, keepdim=True)
        q /= q.sum(-1, keepdim=True)

        counts = torch.zeros(vocab)
        for _ in range(trials):
            draft_tokens = [sample_from(q[i], generator) for i in range(K)]
            tokens, _ = rejection_sample(draft_tokens, q, p, generator)
            counts[tokens[0]] += 1

        empirical = counts / trials
        expected = p[0]
        stderr = (expected * (1 - expected) / trials).sqrt()
        deviation = (empirical - expected).abs()
        worst = float((deviation / stderr).max())

        print(f"K={K}: max |empirical - p| = {float(deviation.max()):.4f} "
              f"({worst:.2f} standard errors)")
        assert worst < 5.0, (
            f"output distribution does not match the target: {worst:.2f} sigma\n"
            f"  expected  {expected.tolist()}\n"
            f"  empirical {empirical.tolist()}"
        )

    print("gate 1 PASSED")


# ---------------------------------------------------------------------------
# gate 2 — greedy exact match against the real target model
# ---------------------------------------------------------------------------

def test_greedy_exact_match(
    target,
    tokenizer,
    prompts: list[str],
    k_values: tuple[int, ...],
    max_new_tokens: int,
    seed: int,
) -> None:
    print("\n== gate 2: speculative greedy == plain greedy, token for token ==")
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    device = next(target.parameters()).device

    drafts = {
        # near-zero acceptance: stresses reject-at-first-token every round
        "random": build_random_draft(target.config, seed=seed),
        # high but partial acceptance: the target drafting for itself at
        # temperature 0.8 diverges from its own argmax often enough to exercise
        # partial acceptance, full acceptance and the bonus token
        "self@0.8": target,
    }
    draft_temperatures = {"random": 1.0, "self@0.8": 0.8}

    failures = 0
    for prompt in prompts:
        prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]
        started = time.perf_counter()
        reference = greedy_generate(target, prompt_ids, max_new_tokens, eos_ids)
        baseline_seconds = time.perf_counter() - started
        print(f"\nprompt: {prompt!r}")
        print(f"  baseline greedy ({len(reference)} tokens, {baseline_seconds:.1f}s): "
              f"{tokenizer.decode(reference)!r}")

        for name, draft in drafts.items():
            for K in k_values:
                stats = SpecStats()
                generator = make_generator(device, seed)
                out = speculative_generate(
                    target, draft, prompt_ids, max_new_tokens, K, eos_ids, generator,
                    temperature=0.0, draft_temperature=draft_temperatures[name], stats=stats,
                )
                ok = out == reference
                failures += not ok
                print(f"  draft={name:9s} K={K}: {'MATCH ' if ok else 'DIFFER'}  {stats}")
                if not ok:
                    first = next((i for i, (a, b) in enumerate(zip(out, reference)) if a != b),
                                 min(len(out), len(reference)))
                    print(f"    first divergence at index {first}")
                    print(f"    reference   {reference[max(0, first - 3):first + 3]}")
                    print(f"    speculative {out[max(0, first - 3):first + 3]}")

    assert failures == 0, f"{failures} speculative runs diverged from greedy decoding"
    print("\ngate 2 PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--k", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--trials", type=int, default=40000, help="samples for the distribution gate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-model", action="store_true", help="run the maths gate only")
    args = parser.parse_args()

    test_rejection_sampling_distribution(trials=args.trials, seed=args.seed)

    if args.skip_model:
        print("\n--skip-model: gate 2 not run")
        return

    model_path = os.path.expanduser(args.model)
    assert os.path.isdir(model_path), f"model not found at {model_path}"

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"\nloading target from {model_path} onto {args.device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    target = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(args.device).eval()

    test_greedy_exact_match(
        target, tokenizer, args.prompts, tuple(args.k), args.max_new_tokens, args.seed,
    )


if __name__ == "__main__":
    main()
