import time
import torch

from typing import Callable
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from common.types import History
from data import ResumableSampler
from common.config.schemas import TrainingConfig
from common.device import supports_bf16
from training.schedule import lr_at_step
from training.loss import batch_loss, dataloader_loss


def _as_eval_loader(dataloader: DataLoader) -> DataLoader:
    """Unshuffled and fixed clone over the same dataset, regardless of epoch or resume offset."""

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
    *,
    start_step: int = 0,
    history: History | None = None,
    on_eval: Callable[[int, float, float], None] | None = None,
    on_log: Callable[[int, float, float], None] | None = None,
    on_checkpoint: Callable[[int, int, History], None] | None = None,
) -> History:
    """
    Trains model on current device, evaluates loss on both dataloaders every 'cfg.eval_freq' optimizer steps,
    learning rate is overwritten on every optimizer step - governed by cfg's lr-related params.
    'model' may be a compiled wrapper, on_checkpoint must close over the raw module.

    Returns:
        dict[str, list[float | int]] containing:
            eval_steps -> list that contains the indices of the optimizer steps in which the model was evaluated (one entry per eval step).
            train_loss -> list that contains the training loss in every eval step (one entry per eval step).
            val_loss -> list that contains the validation loss in every eval step (one entry per eval step).
            lr -> list that contains the global learning rate for each optimizer step (one entry per optimizer step).
            grad_norm -> list that contains the total norm in each optimizer step (one entry per optimizer step).
            log_steps -> list that contains the indices of the optimizer steps in which live step_loss and tokens_per_sec were logged (one entry per log step).
            step_loss -> list that contains the live step loss in each log step (one entry per log step).
            tokens_per_sec -> list that contains the wall-clock tokens/sec throughput in each log step (one entry per log step).

    Note: in cases where the number of batches in train_dataloader is not a multiple of grad_accum_steps the last micro-batch is dropped.
    Note: the 'epoch' passed to on_checkpoint is the epoch the step was taken in, not the epoch to resume into, resume position is derived from optimizer step alone.
    """

    steps_per_epoch = len(train_dataloader) // cfg.grad_accum_steps
    if steps_per_epoch < 1:
        raise ValueError(f"number of steps per epoch must be at least 1, received: len(train_dataloader)={len(train_dataloader)}, grad_accum_steps={cfg.grad_accum_steps}")
    total_steps = steps_per_epoch * cfg.n_epochs
    if cfg.warmup_steps >= total_steps:
        raise ValueError(f"'warmup steps' < 'total optimizer steps' must hold, received: warmup_steps={cfg.warmup_steps}, total_steps={total_steps}")
    if not 0 <= start_step < total_steps:
        raise ValueError(f"0 <= start_step < total_steps must hold, received: start_step={start_step}, total_steps={total_steps}")

    sampler = train_dataloader.sampler
    if not isinstance(sampler, ResumableSampler):
        raise ValueError(f"train_dataloader.sampler must be a ResumableSampler, received: {type(sampler).__name__}")
    batch_size = train_dataloader.batch_size
    if batch_size is None:
        raise ValueError("train_dataloader.batch_size must exist, received: None")

    if start_step == 0 and history is not None:
        raise ValueError("history must be None when start_step is 0")
    if start_step > 0:
        if history is None:
            raise ValueError(f"'history' must not be None when start_step > 0, received: start_step={start_step}")
        if not (len(history["lr"]) == len(history["grad_norm"]) == start_step):
            raise ValueError(f"'len(history['lr']) == len(history['grad_norm']) == start_step' must hold when start_step > 0, received: start_step={start_step}, lens:{len(history['lr'])},{len(history['grad_norm'])}")

    device = next(model.parameters()).device
    autocast_enabled = cfg.use_bf16 and supports_bf16(device)

    if history is None:
        history = {
            "eval_steps": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "grad_norm": [],
            "log_steps": [],
            "step_loss": [],
            "tokens_per_sec": [],
        }

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

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['eval_steps'].append(step)

        if on_eval is not None:
            on_eval(step, train_loss, val_loss)


    loss_acc = torch.zeros((), device=device)
    steps_since_log = 0
    tokens_since_log = 0
    def flush_log() -> None:
        nonlocal steps_since_log, tokens_since_log, last_log_time
        now = time.perf_counter()
        step_loss = (loss_acc / steps_since_log).item()
        tokens_per_sec = tokens_since_log / (now - last_log_time)

        history['log_steps'].append(global_step)
        history['step_loss'].append(step_loss)
        history['tokens_per_sec'].append(tokens_per_sec)

        if on_log is not None:
            on_log(global_step, step_loss, tokens_per_sec)

        loss_acc.zero_()
        steps_since_log = 0
        tokens_since_log = 0
        last_log_time = now

    start_epoch, step_in_epoch = divmod(start_step, steps_per_epoch)
    global_step = start_step
    if start_step == 0:
        record_eval(0)
    last_log_time = time.perf_counter()
    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        first = step_in_epoch if epoch == start_epoch else 0
        sampler.set_epoch(epoch, start_index=(first * cfg.grad_accum_steps * batch_size))
        batches = iter(train_dataloader)
        for _ in range(first, steps_per_epoch):
            lr = lr_at_step(
                global_step, 
                max_lr=cfg.max_lr, 
                min_lr=cfg.min_lr,
                warmup_steps=cfg.warmup_steps,
                total_steps=total_steps,
            )
            history['lr'].append(lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            for _ in range(cfg.grad_accum_steps):
                x,y = next(batches)
                x,y = x.to(device, non_blocking=x.is_pinned()), y.to(device, non_blocking=y.is_pinned())

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    loss = batch_loss(model, x, y) / cfg.grad_accum_steps
                loss.backward()
                loss_acc += loss.detach()
                tokens_since_log += y.numel()

            total_norm = nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grads_with_norm_(model.parameters(), cfg.grad_clip_norm, total_norm)
            history['grad_norm'].append(total_norm.item())

            optimizer.step()
            global_step += 1
            steps_since_log += 1
            if global_step % cfg.log_freq == 0:
                flush_log()
            if global_step % cfg.eval_freq == 0:
                record_eval(global_step)
            if on_checkpoint is not None and global_step % cfg.checkpoint_freq == 0:
                on_checkpoint(global_step, epoch, history)

    if steps_since_log > 0:
        flush_log()
    if global_step % cfg.eval_freq != 0:
        record_eval(global_step)


    return history
