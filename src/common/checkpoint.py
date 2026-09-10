import torch
import os

from torch import nn
from pathlib import Path
from torch.optim import Optimizer
from common.config.schemas import RunConfig


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    global_step: int,
    epoch: int,
    run_cfg: RunConfig,
    history: dict | None = None,
) -> None:
    """
    Atomically saves a checkpoint to 'path' (parent dirs get created).

    Note: The run config is stored as a JSON-mode dump plus its class name, never as a pickled object.
    Note: 'model' expects the original module, rather than a wrapped one (e.g the one created by 'torch.compile(model)')
    """

    if hasattr(model, "_orig_mod"):
        raise ValueError(f"model must be the original module, not a compiled wrapper, received: {type(model).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(
            {
                "run_cfg_class": type(run_cfg).__name__,
                "run_cfg": run_cfg.model_dump(mode="json"),
                "model_state_dict": model.state_dict(),
                "optimizer_class": type(optimizer).__name__,
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "history": history,
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
    map_location: str | torch.device = "cpu",
) -> dict:
    """
    Restores 'model' (and 'optimizer') in place from the checkpoint at 'path',
    and returns the rest of it: run config as a dump plus class name, global_step, epoch, history.

    Note: optimizer's saved hyperparameters override the original ones.
    Note: 'model' expects the original module, rather than a wrapped one (e.g the one created by 'torch.compile(model)')
    """

    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint found at path: {path}")
    if hasattr(model, "_orig_mod"):
        raise ValueError(f"model must be the original module, not a compiled wrapper, received: {type(model).__name__}")

    checkpoint: dict = torch.load(path, map_location=map_location)

    required = {"run_cfg_class", "run_cfg", "model_state_dict", "global_step", "epoch"}
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

    return checkpoint
