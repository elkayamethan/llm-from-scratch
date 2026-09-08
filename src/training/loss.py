import torch

from torch import Tensor, nn
from torch.utils.data import DataLoader
from itertools import islice


def batch_loss(model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    logits = model(x)
    return nn.functional.cross_entropy(logits.flatten(0,1), y.flatten())


@torch.no_grad()
def dataloader_loss(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    max_batches: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    """Evaluates loss across dataloader.
    
    Note: loss is measured under autocast_dtype, if 'None' then defaults to the dtype the model is in."""

    if max_batches is not None and max_batches <= 0:
        raise ValueError(f"max_batches must be > 0 or None, received: {max_batches}")

    device = next(model.parameters()).device

    is_training = model.training
    model.eval()
    try:
        loss_sum = torch.zeros((), device=device)
        tokens_sum = 0
        with torch.autocast(device_type=device.type, dtype=autocast_dtype or torch.float32, enabled=autocast_dtype is not None):
            for x,y in islice(dataloader, max_batches):
                x = x.to(device, non_blocking=x.is_pinned())
                y = y.to(device, non_blocking=y.is_pinned())
                loss_sum += batch_loss(model, x, y) * y.numel()
                tokens_sum += y.numel()

        if tokens_sum == 0:
            raise ValueError("dataloader yielded no batches")
        loss = loss_sum.item() / tokens_sum

    finally:
        model.train(is_training)

    return loss
