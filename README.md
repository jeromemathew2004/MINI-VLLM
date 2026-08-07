# mini-vllm

mini-vllm is a compact, LLM inference engine for exploring modern LLM serving techniques from first principles. Its design is intentionally small and transparent, making it ideal for studying the core mechanics of generation: scheduling, paged KV-cache allocation, batched execution, and sampling.

## What it supports

- Native support for Qwen3 as the primary model family.
- Gemma3 support already present in the model tree.
- Continuous batching for higher throughput under load.
- Paged KV-cache management with prefix caching support.
- CUDA graph execution for lower dispatch overhead.
- Speculative decoding with two interchangeable proposers: a lightweight draft model or an n-gram lookup that requires no extra model at all.

## Requirements

- Python 3.10+ is recommended; the current environment is running Python 3.13.3.
- A CUDA-capable NVIDIA GPU is required for the default runtime path.
- Install dependencies from [requirements.txt](requirements.txt).
- The demo scripts expect a local model at `~/huggingface/Qwen3-0.6B/` unless you provide another path explicitly.
- `mini-flash-attention` is pulled from GitHub rather than PyPI, reflecting its specialized backend requirements.

## Installation

A polished setup on Windows PowerShell looks like this:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
hf download Qwen/Qwen3-0.6B
```

A quick end-to-end health check is:

```sh
python experiments/engine_spec_gate.py --compare --cuda-graph
```

## Quick start

Launch an interactive session:

```sh
python chat.py --model ~/huggingface/Qwen3-0.6B/
```

The chat interface supports these commands:

- `/help` shows available commands.
- `/clear` clears the conversation history.
- `/history` prints the current conversation.
- `/system <text>` updates the system prompt.
- `/maxtokens <num>` changes the generation limit.
- `/top-k <num>` adjusts top-k sampling.
- `/top-p <float>` adjusts nucleus sampling.
- `/temperature <float>` changes sampling temperature.
- `/settings` prints the current configuration.
- `/exit` closes the session.

Run a scripted inference demo:

```sh
python run.py
```

Benchmark the implementation:

```sh
python benchmark/run_mini_vllm.py
```

Compare it against vLLM:

```sh
python benchmark/run_vllm.py
```

## Speculative decoding

Speculative decoding preserves the target model’s output distribution while allowing a cheaper proposer to guess several tokens ahead. A proposer emits a short candidate prefix, the target evaluates it in a single forward pass, and a rejection sampler keeps the longest prefix the target would have produced anyway.

The engine currently supports two proposers:

- `"draft"`: a compact draft model is run repeatedly to propose candidate tokens.
- `"ngram"`: the most recent tokens are matched against the request’s own history and proposed without any additional model.

A minimal configuration looks like this:

```python
from minivllm.config.config import Config

Config(
    model="~/huggingface/Qwen3-0.6B/",
    use_speculative_decoding=True,
    speculative_method="ngram",
    num_speculative_tokens=4,
)
```

The n-gram proposer is especially effective on repetitive prompts, and the benchmark sweep captures the trade-off between throughput and acceptance rate:

```sh
python benchmark/spec_sweep.py
```

More detail, measurement notes, and implementation history are available in [PROGRESS.md](PROGRESS.md) and [docs/future_upgrades.md](docs/future_upgrades.md).

## Testing

Run the regression and GPU validation suite with:

```sh
pytest tests/ -m "not gpu"
pytest tests/
pytest tests/ --slow
```

## Windows and GPU notes

- On Windows, `use_cuda_graph=True` may require `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0`.
- The default configuration can exceed VRAM on smaller GPUs; the 4 GB profile in [experiments/engine_spec_gate.py](experiments/engine_spec_gate.py) is a safer starting point.
- If you rebuild `mini-flash-attention`, re-apply the patch in [patches/mini-flash-attention-decode-race.patch](patches/mini-flash-attention-decode-race.patch) to keep the decode path deterministic.

## Project layout

- [minivllm/engine](minivllm/engine) contains the high-level engine and request lifecycle.
- [minivllm/executor](minivllm/executor) handles model execution, KV-cache usage, and speculative decoding.
- [minivllm/scheduler](minivllm/scheduler) implements the continuous-batching scheduler.
- [minivllm/models](minivllm/models) contains the supported model implementations.
- [tests](tests) includes correctness and end-to-end checks for the engine.

## References

This project was developed with inspiration from:

- [vLLM](https://github.com/vllm-project/vllm)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
