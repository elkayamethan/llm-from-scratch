import torch

from inference.sampling import top_k_filter, apply_temperature
from torch import Tensor, nn


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: Tensor,
    context_size: int,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> Tensor:
    """Autoregressively generates tokens past batched 'idx' by up to 'max_new_tokens' tokens
    
    Always returns a rectangular output (sequences that get EOS early are padded with EOS)"""

    is_training = model.training
    model.eval()

    try:
        unfinished = torch.ones((idx.shape[0], 1), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            curr_input = idx[..., -context_size:]

            logits: Tensor = model(curr_input)
            logits = logits[..., -1, :]
            logits = apply_temperature(logits, temperature)
            logits = top_k_filter(logits, top_k)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if eos_id is not None:
                next_id = torch.where(unfinished, next_id, eos_id)
                unfinished &= next_id != eos_id

            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and not torch.any(unfinished):
                break
            
    finally:
        model.train(is_training)

    return idx
