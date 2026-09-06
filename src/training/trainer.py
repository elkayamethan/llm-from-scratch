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

    eval_dataloader = DataLoader(
        dataloader.dataset, 
        shuffle=False, 
        batch_size=dataloader.batch_size, 
        num_workers=dataloader.num_workers, 
        pin_memory=dataloader.pin_memory
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
    learning rate is overwritten on every step - governed by cfg's lr-related params.

    Returns:
        dict[str, list[float | int]] containing:
            eval_steps -> list that contains the indices of the steps in which the model was evaluated (one entry per eval step).
            train_loss -> list that contains the training loss in every eval step (one entry per eval step).
            val_loss -> list that contains the validation loss in every eval step (one entry per eval step).
            lr -> list that contains the global learning rate in each global step (one entry per optimizer step).
            grad_norm -> list that contains 'torch.nn.utils.get_total_norm' in each global step (one entry per optimizer step).
    """

    total_steps = len(train_dataloader) * cfg.n_epochs
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
        train_loss = dataloader_loss(model, eval_train_dataloader, max_batches=cfg.eval_batches)
        val_loss = dataloader_loss(model, eval_val_dataloader, max_batches=cfg.eval_batches)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        eval_steps.append(step)

        if on_eval is not None:
            on_eval(step, train_loss, val_loss)


    global_step = 0
    record_eval(global_step)
    for _ in range(cfg.n_epochs):
        model.train()
        for x,y in train_dataloader:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = batch_loss(model, x, y)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
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
