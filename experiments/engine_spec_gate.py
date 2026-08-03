"""Greedy exact-match gate for the engine, on a 4 GB GPU.

This is the engine-level counterpart to the Phase 2 prototype's gate 2. Phase 3
adds no behaviour — the draft model is loaded and then ignored — so the two
invocations below must emit byte-identical greedy token ids:

    python experiments/engine_spec_gate.py                  # baseline
    python experiments/engine_spec_gate.py --spec           # draft loaded
    python experiments/engine_spec_gate.py --compare        # run both, diff them

Once Phase 4/5 land, `--spec` starts actually drafting and verifying, and this
same comparison becomes the Phase 6 correctness gate unchanged: rejection
sampling is distribution-preserving, so greedy output must still match exactly.

Token ids are written to JSON rather than eyeballed, because "looks the same"
is not the property under test.

Windows note: CUDA-graph capture needs TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
(torch's static launcher passes a 64-bit device pointer through a C long, which
is 32-bit on Windows). This script sets it before importing torch.
"""

import argparse
import gc
import json
import logging
import os
import sys

# Must precede the torch import: inductor reads this at config-module import.
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minivllm.config.config import Config  # noqa: E402
from minivllm.config.sampling import SamplingParams  # noqa: E402
from minivllm.engine.engine import Engine  # noqa: E402

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")

TARGET = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-draft-random/")

PROMPTS = [
    "What is the meaning of life?",
    "How do I get started with LLMs?",
]


def build_config(spec: bool, cuda_graph: bool) -> Config:
    # Tuned for a 4 GB RTX 3050. The stock kv_cache_block_size of 256 costs
    # 28 MiB per block for Qwen3-0.6B (2 * 28 layers * 256 * 8 kv heads *
    # 128 head dim * 2 bytes), which is far too coarse a granularity when
    # roughly 800 MiB of the 4 GB is already gone to the CUDA context and the
    # Windows desktop. 64 brings that to 7 MiB.
    return Config(
        model=TARGET,
        max_model_len=1024,
        max_num_batched_tokens=2048,
        max_num_batched_seqs=8,
        kv_cache_block_size=64,
        gpu_memory_utilization=0.9,
        use_cuda_graph=cuda_graph,
        use_speculative_decoding=spec,
        draft_model=DRAFT if spec else "",
    )


def run(spec: bool, cuda_graph: bool, max_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)
        for p in PROMPTS
    ]

    engine = Engine(build_config(spec, cuda_graph))

    # temperature 1.0 with top_k 0 and top_p 1.0 sends Sampler down its argmax
    # branch, so this is deterministic greedy and never imports flashinfer
    # (which has no reliable Windows wheel).
    sp = SamplingParams(temperature=1.0, top_k=0, top_p=1.0,
                        max_tokens=max_tokens)
    outputs = engine.generate(prompts, sp)

    result = [{"prompt": o["prompt"], "completion": o["completion"],
               "token_ids": tokenizer.encode(o["completion"])} for o in outputs]

    # --compare builds a second engine straight after this one. nn.Module
    # graphs contain reference cycles, so refcounting alone will not free the
    # weights or the KV cache; without the collect the second engine sees the
    # first one's memory still resident and sizes its cache down (or trips
    # the kv_cache_num_blocks assert) on a 4 GB card.
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return result


def diff(base, spec) -> bool:
    ok = len(base) == len(spec)
    for i, (x, y) in enumerate(zip(base, spec)):
        same = x["token_ids"] == y["token_ids"]
        ok &= same
        print(f"[{i}] n_tokens={len(x['token_ids']):4d}  identical={same}")
        if not same:
            for j, (p, q) in enumerate(zip(x["token_ids"], y["token_ids"])):
                if p != q:
                    print(f"     first divergence at token {j}: {p} != {q}")
                    break
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="store_true",
                    help="load the draft model (use_speculative_decoding=True)")
    ap.add_argument("--compare", action="store_true",
                    help="run baseline and --spec in one process and diff them")
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.compare:
        # Note both engines are built in one process; the first is garbage
        # collected only when it goes out of scope, so on a 4 GB card this
        # needs the two runs to be sequential, not concurrent.
        base = run(False, args.cuda_graph, args.max_tokens)
        spec = run(True, args.cuda_graph, args.max_tokens)
        ok = diff(base, spec)
        print("\nGATE:", "PASS - byte-identical" if ok else "FAIL - output diverged")
        return 0 if ok else 1

    result = run(args.spec, args.cuda_graph, args.max_tokens)
    out = args.out or ("spec.json" if args.spec else "base.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")
    for r in result:
        print("-" * 60)
        print(r["completion"][:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
