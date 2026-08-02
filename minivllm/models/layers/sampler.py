import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                logits: torch.Tensor,
                temperatures: torch.Tensor|None,
                top_p: torch.Tensor|None,
                top_k: torch.Tensor|None) -> torch.Tensor:

        if temperatures is not None:
            logits = logits / temperatures.unsqueeze(-1)
        
        if top_p is not None or top_k is not None:
            # imported lazily: flashinfer is only needed for top-k/top-p, it is
            # absent from requirements.txt, and it has no reliable Windows
            # wheel. Greedy decoding — which the speculative-decoding
            # correctness harness runs on — must not depend on it.
            from flashinfer.sampling import top_k_top_p_sampling_from_logits

            # flashinfer has a bug when default device is cuda
            # see https://github.com/flashinfer-ai/flashinfer/issues/2333
            device = torch.get_default_device()
            torch.set_default_device("cpu")
            sampled_tokens = top_k_top_p_sampling_from_logits(
                logits,
                top_k,
                top_p,
            )
            torch.set_default_device(device)
            return sampled_tokens
        
        return torch.argmax(logits, dim=-1)
