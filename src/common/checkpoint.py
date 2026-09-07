import torch
import os

from torch import nn
from pathlib import Path
from torch.optim import Optimizer
from common.config.schemas import TransformerConfig, TrainingConfig


def save_checkpoint(
    path: Path, 
    *, 
    model: nn.Module, 
    optimizer: Optimizer, 
    global_step: int, 
    epoch: int,
    model_cfg: TransformerConfig, 
    train_cfg: TrainingConfig,
    generator: torch.Generator | None = None,
    history: dict | None = None,
) -> None:
    """Atomically saves a checkpoint to 'path' (parent dirs get created).
    Note: Configs are stored as dumps + class name"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(
            {
                "model_cfg_class": type(model_cfg).__name__,
                "model_cfg": model_cfg.model_dump(),
                "model_state_dict": model.state_dict(),

                "train_cfg": train_cfg.model_dump(),
                "optimizer_class": type(optimizer).__name__,
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "history": history,

                "generator_state": generator.get_state() if generator is not None else None,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    finally:
        tmp_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    generator: torch.Generator | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Restores 'model' (and 'optimizer'/'generator') in place from the checkpoint at 'path',
    and returns the rest of it: configs as dumps, global_step, epoch, history.
    Note: optimizer's saved hyperparameters override the original ones"""

    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint found at path: {path}")

    checkpoint: dict = torch.load(path, map_location=map_location)

    required = {"model_cfg_class", "model_cfg", "model_state_dict", "global_step", "epoch"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Not a valid checkpoint, missing keys: {sorted(missing)}, path: {path}")

    model.load_state_dict(checkpoint.pop("model_state_dict"))

    if optimizer is not None:
        optimizer_state = checkpoint.pop("optimizer_state_dict", None)
        saved_class = checkpoint.get("optimizer_class")
        if saved_class != type(optimizer).__name__:
            raise ValueError(f"Optimizer mismatch, checkpoint was written by {saved_class}, received: {type(optimizer).__name__}")
        if optimizer_state is None:
            raise ValueError(f"Checkpoint holds no optimizer state, path: {path}")
        optimizer.load_state_dict(optimizer_state)

    if generator is not None:
        generator_state = checkpoint.pop("generator_state", None)
        if checkpoint.get("generator_state") is None:
            raise ValueError(f"Checkpoint holds no generator state, path: {path}")
        generator.set_state(generator_state)

    return checkpoint
