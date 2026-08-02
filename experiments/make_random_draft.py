"""Write a randomly-initialised tiny Qwen3 to disk for use as a draft model.

Phase 1 of speculative-decoding-plan.md — training a real draft — is deferred
until correctness is locked, because rejection sampling is distribution-
preserving for any draft: draft quality moves the acceptance rate and nothing
else. Phase 2 confirmed that empirically (a random draft hit 0% acceptance and
still reproduced greedy output exactly). So the engine work needs *a* draft
checkpoint on disk, not a good one.

The checkpoint is deliberately smaller than the 10-30M non-embedding parameters
the plan specifies for the trained draft: this one exists to make the
correctness harness run fast, not to be accepted often.

Usage:
    python experiments/make_random_draft.py
    python experiments/make_random_draft.py --hidden-size 512 --num-layers 6
"""

import argparse
import os
import shutil

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

DEFAULT_TARGET = "~/huggingface/Qwen3-0.6B/"
DEFAULT_OUTPUT = "~/huggingface/Qwen3-draft-random/"

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET, help="target model, for vocab and RoPE settings")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    target_path = os.path.expanduser(args.target)
    output_path = os.path.expanduser(args.output)
    assert os.path.isdir(target_path), f"target model not found at {target_path}"

    target_config = AutoConfig.from_pretrained(target_path)

    config = Qwen3Config(
        # The vocabulary must match the target exactly — rejection sampling
        # compares p(x) and q(x) for the same token id. Config.__post_init__
        # asserts this again at load time.
        vocab_size=target_config.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        max_position_embeddings=target_config.max_position_embeddings,
        rope_theta=target_config.rope_theta,
        rms_norm_eps=target_config.rms_norm_eps,
        # Tied, or the ~152k-row LM head alone would dominate the checkpoint.
        # minivllm's Qwen3ForCausalLM honours this: it aliases lm_head.weight
        # onto the embedding and its load_weights skips lm_head.weight.
        tie_word_embeddings=True,
        bos_token_id=target_config.bos_token_id,
        eos_token_id=target_config.eos_token_id,
        dtype=torch.bfloat16,
    )

    torch.manual_seed(args.seed)
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)

    embedding_params = config.vocab_size * config.hidden_size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"draft model: {total_params / 1e6:.1f}M parameters total, "
          f"{(total_params - embedding_params) / 1e6:.1f}M non-embedding")

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)

    # Copy the target's tokenizer alongside so the checkpoint is self-contained
    # and the shared vocabulary is verifiable from the directory itself.
    for name in TOKENIZER_FILES:
        source = os.path.join(target_path, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(output_path, name))

    print(f"written to {output_path}")

    # The engine dispatches on architectures[0] via the _MODELS registry, and
    # loads weights by HF parameter name. Check both here rather than at engine
    # startup, where CUDA and the flash-attention backend would be needed too.
    written = AutoConfig.from_pretrained(output_path)
    assert written.architectures == ["Qwen3ForCausalLM"], written.architectures
    assert written.vocab_size == target_config.vocab_size

    tokenizer = AutoTokenizer.from_pretrained(output_path)
    target_tokenizer = AutoTokenizer.from_pretrained(target_path)
    probe = "The capital of France is"
    assert tokenizer(probe)["input_ids"] == target_tokenizer(probe)["input_ids"]

    print("verified: architecture registered as Qwen3ForCausalLM, vocab and tokenizer match the target")


if __name__ == "__main__":
    main()
