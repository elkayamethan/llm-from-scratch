from math import cos, pi

def lr_at_step(
    step: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Returns the learning rate for a given optimizer step (0-indexed), warms up linearly and then applies cosine decay"""

    if step >= total_steps:
        return min_lr
    if step < warmup_steps:
        return (max_lr * (step + 1)) / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
