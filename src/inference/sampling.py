import torch
from torch import Tensor

def top_k_filter(logits: Tensor, k: int | None) -> Tensor:
    """Apply top-k masking to logits"""

    if k is None or k >= logits.shape[-1]:
        return logits

    top_logits, _ = torch.topk(logits, k)
    threshold = top_logits[..., -1:]
    return torch.where(logits < threshold, float('-inf'), logits)


def apply_temperature(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Apply temperature scaling to logits"""

    if temperature < 0:
        raise ValueError(f"Temperature must be >= 0, received: {temperature}")
    if temperature == 0:
        return torch.where(logits == logits.max(dim=-1, keepdim=True).values, logits, float('-inf'))

    return logits / temperature
