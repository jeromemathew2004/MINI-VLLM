"""Pytest configuration for the whole repo.

Its existence at the root is load-bearing, not incidental: pytest inserts the
directory holding the rootmost conftest into `sys.path`, which is what makes
`import minivllm` and `import tests.spec_harness` work regardless of how pytest
was invoked. Without it, `pytest tests/` and `python -m pytest tests/` resolve
imports differently — the latter works only because it happens to prepend the
working directory.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Must be set before torch is imported. On Windows, CUDA-graph capture dies with
# "OverflowError: Python int too large to convert to C long" without it: torch's
# static launcher passes a 64-bit device pointer through a C long, which is
# 32-bit there. Harmless elsewhere.
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False,
        help="also run tests that build engines and generate text (minutes, and "
             "they load model checkpoints)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: needs a CUDA device and the Qwen3-0.6B checkpoint")
    config.addinivalue_line(
        "markers", "slow: builds engines and generates; opt in with --slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
