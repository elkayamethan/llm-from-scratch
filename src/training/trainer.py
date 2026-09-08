import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from common.config.schemas import TrainingConfig
from common.device import supports_bf16
from training.schedule import lr_at_step
from training.loss import batch_loss, dataloader_loss
from typing import Callable


def _as_eval_loader(dataloader: DataLoader) -> DataLoader:
    """To avoid drawing from a dataloader's generator"""

    batch_size = dataloader.batch_size or getattr(dataloader.batch_sampler, "batch_size", None)
    if batch_size is None:
        raise ValueError("Dataloader must have a batch_size")

    eval_dataloader = DataLoader(
        dataloader.dataset, 
        shuffle=False, 
        batch_size=batch_size,
        num_workers=dataloader.num_workers, 
        pin_memory=dataloader.pin_memory,
    )

    return eval_dataloader


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    cfg: TrainingConfig,
    on_eval: Callable[[int, float, float], None] | None = None,
) -> dict[str, list[float | int]]:
    """
    Trains model on current device, evaluates loss on both dataloaders every 'cfg.eval_freq' optimizer steps,
    learning rate is overwritten on every optimizer step - governed by cfg's lr-related params.

    Returns:
        dict[str, list[float | int]] containing:
            eval_steps -> list that contains the indices of the steps in which the model was evaluated (one entry per eval step).
            train_loss -> list that contains the training loss in every eval step (one entry per eval step).
            val_loss -> list that contains the validation loss in every eval step (one entry per eval step).
            lr -> list that contains the global learning rate in each global step (one entry per optimizer step).
            grad_norm -> list that contains the total norm in each global step (one entry per optimizer step).
    
    Note: in cases where the number of batches in train_dataloader is not a multiple of grad_accum_steps the last micro-batch is dropped.
    """

    steps_per_epoch = len(train_dataloader) // cfg.grad_accum_steps
    if steps_per_epoch < 1:
        raise ValueError(f"number of steps per epoch must be at least 1, received: len(train_dataloader)={len(train_dataloader)}, grad_accum_steps={cfg.grad_accum_steps}")
    total_steps = steps_per_epoch * cfg.n_epochs
    if cfg.warmup_steps >= total_steps:
        raise ValueError(f"'warmup steps' < 'total optimizer steps' must hold, received: warmup_steps={cfg.warmup_steps}, total_steps={total_steps}")

    device = next(model.parameters()).device
    autocast_enabled = cfg.use_bf16 and supports_bf16(device)

    eval_steps = []
    train_losses = []
    val_losses = []
    lrs = []
    grad_norms = []


    eval_train_dataloader = _as_eval_loader(train_dataloader)
    eval_val_dataloader = _as_eval_loader(val_dataloader)
    def record_eval(step: int) -> None:
        train_loss = dataloader_loss(
            model,
            eval_train_dataloader, 
            max_batches=cfg.eval_batches,
            autocast_dtype=torch.bfloat16 if autocast_enabled else None,
        )
        val_loss = dataloader_loss(
            model,
            eval_val_dataloader, 
            max_batches=cfg.eval_batches,
            autocast_dtype=torch.bfloat16 if autocast_enabled else None,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        eval_steps.append(step)

        if on_eval is not None:
            on_eval(step, train_loss, val_loss)


    global_step = 0
    record_eval(global_step)
    for _ in range(cfg.n_epochs):
        model.train()
        batches = iter(train_dataloader)
        for _ in range(steps_per_epoch):
            lr = lr_at_step(
                global_step, 
                max_lr=cfg.max_lr, 
                min_lr=cfg.min_lr,
                warmup_steps=cfg.warmup_steps,
                total_steps=total_steps,
            )
            lrs.append(lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            for _ in range(cfg.grad_accum_steps):
                x,y = next(batches)
                x,y = x.to(device, non_blocking=x.is_pinned()), y.to(device, non_blocking=y.is_pinned())

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    loss = batch_loss(model, x, y) / cfg.grad_accum_steps
                loss.backward()

            total_norm = nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grads_with_norm_(model.parameters(), cfg.grad_clip_norm, total_norm)
            grad_norms.append(total_norm.item())

            optimizer.step()
            global_step += 1
            if global_step % cfg.eval_freq == 0:
                record_eval(global_step)

    if global_step % cfg.eval_freq != 0:
        record_eval(global_step)


    history = {
        "eval_steps": eval_steps,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "lr": lrs,
        "grad_norm": grad_norms,
    }
    return history
