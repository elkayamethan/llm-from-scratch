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
) -> float:

    if max_batches is not None and max_batches <= 0:
        raise ValueError(f"max_batches must be > 0 or None, received: {max_batches}")
    
    device = next(model.parameters()).device

    is_training = model.training
    model.eval()
    try:
        loss_sum = 0
        tokens_sum = 0
        for x,y in islice(dataloader, max_batches):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss_sum += batch_loss(model, x, y).item() * y.numel()
            tokens_sum += y.numel()

        if tokens_sum == 0:
            raise ValueError("dataloader yielded no batches")
        loss = loss_sum / tokens_sum

    finally:
        model.train(is_training)

    return loss
