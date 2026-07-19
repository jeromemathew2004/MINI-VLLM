# mini-vllm

mini-vllm is a lightweight LLM inference engine built from scratch for learning and experimentation. The implementation stays intentionally small so the main execution flow is easy to inspect.

## What It Supports

- Qwen3 models are the primary target today.
- Gemma3 support is also present in the model tree.
- Continuous batching for improved throughput.
- CUDA graph execution.
- KV cache management for generation.
- Prefix caching.

## Requirements

- Python 3.10+ is recommended.
- A CUDA-capable GPU is required for the current setup.
- Install the project dependencies from [requirements.txt](requirements.txt).

## Install

Create an environment, install dependencies, and download a compatible model:

```sh
pip install -r requirements.txt
hf download Qwen/Qwen3-0.6B
```

The demo scripts expect the model to be available locally. By default they look for `~/huggingface/Qwen3-0.6B/`.
One dependency, `mini-flash-attention`, is pulled from GitHub because it is not published on PyPI.

## Run

Interactive chat:

```sh
python chat.py --model ~/huggingface/Qwen3-0.6B/
```

The chat session supports these commands:

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

Scripted inference demo:

```sh
python run.py
```

Benchmark the mini implementation:

```sh
python benchmark/run_mini_vllm.py
```

Compare against vLLM:

```sh
python benchmark/run_vllm.py
```

The benchmark scripts load repeated prompts and report throughput in tokens per second, which makes it easy to compare this implementation with vLLM on the same model.

## Notes

- The scripts use Hugging Face chat templates and `enable_thinking=True` for Qwen3-style prompts.
- If you use a different model, update the local model path in the scripts or pass it to `chat.py` with `--model`.

## References

This project was developed with inspiration from:

- [vLLM](https://github.com/vllm-project/vllm)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
