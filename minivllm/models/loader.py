import logging
import os
from glob import glob
import torch
import safetensors
from transformers import PretrainedConfig

from minivllm.config.config import Config
from minivllm.models.models import get_model_class

def _initialize_model(hf_config: PretrainedConfig) -> torch.nn.Module:
    architecture = hf_config.architectures[0]
    cls = get_model_class(architecture)
    if cls is None:
        raise ValueError(f"Model architecture {architecture} is not supported.")
    model = cls(hf_config)
    return model

def _get_weights_iterator(path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safetensors.safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def _load(path: str, hf_config: PretrainedConfig) -> torch.nn.Module:
    model = _initialize_model(hf_config)
    model.load_weights(_get_weights_iterator(path))
    return model


def load_model(config: Config) -> torch.nn.Module:
    logging.info("Loading model on device...")
    return _load(config.model, config.hf_config)


def load_draft_model(config: Config) -> torch.nn.Module:
    """Load the speculative-decoding draft model.

    Goes through the same registry and the same `load_weights` path as the
    target, so any architecture in `_MODELS` can serve as a draft as long as it
    shares the target's vocabulary (checked in `Config.__post_init__`).
    """
    logging.info("Loading draft model on device...")
    return _load(config.draft_model, config.draft_hf_config)
