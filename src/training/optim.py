from torch import nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Returns 2 optimizer parameter groups:

    One with weight_decay for dim >= 2 (e.g: weight matrices),
    The other without weight_decay for dim < 2 (e.g: layernorm)."""

    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, received: {weight_decay}")

    decaying = []
    non_decaying = []
    for p in model.parameters():
        if p.requires_grad:
            if p.dim() >= 2:
                decaying.append(p)
            else:
                non_decaying.append(p)

    groups = [
        {"params": decaying, "weight_decay": weight_decay},
        {"params": non_decaying, "weight_decay": 0.0}
    ]

    return groups
