from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

from architecture.models import GPT
from common.checkpoint import load_checkpoint, save_checkpoint
from common.config.schemas import GPTConfig, RunConfig, TrainingConfig
from data import PretrainingDataset
from data.loader import create_pretraining_dataloaders
from training import param_groups, train
from training.loss import batch_loss


class _Recording(Dataset):
    """Records every index read so two runs' consumption can be compared"""

    def __init__(self, inner: Dataset) -> None:
        self.inner = inner
        self.seen: list[int] = []

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen.append(index)
        return self.inner[index]

    def __len__(self) -> int:
        return len(self.inner) # type: ignore


class _Stop(Exception):
    pass


def _loaders(cfg: TrainingConfig) -> tuple:
    g = torch.Generator().manual_seed(0)
    train_ds = _Recording(PretrainingDataset(torch.randint(0, 64, (2000,), generator=g), context_size=16, stride=16))
    val_ds = PretrainingDataset(torch.randint(0, 64, (500,), generator=g), context_size=16, stride=16)
    return train_ds, create_pretraining_dataloaders(train_ds, val_ds, batch_size=cfg.batch_size, seed=5) # type: ignore


def _model_and_opt(mcfg: GPTConfig, tcfg: TrainingConfig) -> tuple[GPT, AdamW]:
    torch.manual_seed(1)
    model = GPT(mcfg)
    opt = AdamW(param_groups(model, tcfg.weight_decay), lr=tcfg.max_lr, betas=tcfg.betas, eps=tcfg.eps)
    return model, opt


def test_history_shape(tiny_model_cfg, tiny_train_cfg) -> None:
    _, (tl, vl) = _loaders(tiny_train_cfg)
    model, opt = _model_and_opt(tiny_model_cfg, tiny_train_cfg)
    h = train(model, tl, vl, opt, tiny_train_cfg)
    assert len(h["lr"]) == len(h["grad_norm"]) == 30
    assert h["eval_steps"] == [0, 10, 20, 30]
    assert h["log_steps"] == [4, 8, 12, 16, 20, 24, 28, 30]
    assert len(h["step_loss"]) == len(h["tokens_per_sec"]) == 8
    assert all(v > 0 for v in h["tokens_per_sec"])
    assert all(0 < v < 10 for v in h["step_loss"])


def test_resume_is_exact(tmp_path: Path, tiny_model_cfg, tiny_train_cfg) -> None:
    ds_a, (tl_a, vl_a) = _loaders(tiny_train_cfg)
    model_a, opt_a = _model_and_opt(tiny_model_cfg, tiny_train_cfg)
    h_a = train(model_a, tl_a, vl_a, opt_a, tiny_train_cfg)

    ckpt = tmp_path / "step_000020.pt"
    ds_b, (tl_b, vl_b) = _loaders(tiny_train_cfg)
    model_b, opt_b = _model_and_opt(tiny_model_cfg, tiny_train_cfg)
    run_cfg = RunConfig(run_name="t", seed=5, data_dir=tmp_path, checkpoint_dir=tmp_path,
                        model=tiny_model_cfg, training=tiny_train_cfg)

    def save_and_crash(step: int, epoch: int, history: dict) -> None:
        save_checkpoint(ckpt, model=model_b, optimizer=opt_b, global_step=step, epoch=epoch,
                        run_cfg=run_cfg, history=history)
        raise _Stop

    with pytest.raises(_Stop):
        train(model_b, tl_b, vl_b, opt_b, tiny_train_cfg, on_checkpoint=save_and_crash)

    model_c, opt_c = _model_and_opt(tiny_model_cfg, tiny_train_cfg)
    state = load_checkpoint(ckpt, model=model_c, optimizer=opt_c)
    assert state["global_step"] == 20 and state["epoch"] == 1
    h_c = train(model_c, tl_b, vl_b, opt_c, tiny_train_cfg, start_step=20, history=state["history"])

    assert ds_b.seen == ds_a.seen
    for (name, a), (_, c) in zip(model_a.named_parameters(), model_c.named_parameters()):
        assert torch.equal(a, c), name
    for key in ("eval_steps", "train_loss", "val_loss", "lr", "grad_norm", "log_steps", "step_loss"):
        assert h_c[key] == h_a[key], key


def test_start_step_requires_matching_history(tiny_model_cfg, tiny_train_cfg) -> None:
    _, (tl, vl) = _loaders(tiny_train_cfg)
    model, opt = _model_and_opt(tiny_model_cfg, tiny_train_cfg)
    with pytest.raises(ValueError, match="history"):
        train(model, tl, vl, opt, tiny_train_cfg, start_step=3)
    bad = {k: [] for k in ("eval_steps", "train_loss", "val_loss", "lr", "grad_norm", "log_steps", "step_loss", "tokens_per_sec")}
    with pytest.raises(ValueError, match="start_step"):
        train(model, tl, vl, opt, tiny_train_cfg, start_step=3, history=bad)
    with pytest.raises(ValueError, match="start_step"):
        train(model, tl, vl, opt, tiny_train_cfg, start_step=30)


def test_requires_resumable_sampler(tiny_model_cfg, tiny_train_cfg) -> None:
    from torch.utils.data import DataLoader
    ds, (tl, vl) = _loaders(tiny_train_cfg)
    model, opt = _model_and_opt(tiny_model_cfg, tiny_train_cfg)
    with pytest.raises(ValueError, match="ResumableSampler"):
        train(model, DataLoader(ds, batch_size=4, shuffle=True), vl, opt, tiny_train_cfg)


def test_grad_accum_matches_large_batch(tiny_model_cfg) -> None:
    torch.manual_seed(0)
    model = GPT(tiny_model_cfg)
    x = torch.randint(0, 64, (8, 16))
    y = torch.randint(0, 64, (8, 16))
    batch_loss(model, x, y).backward()
    reference = [p.grad.clone() for p in model.parameters()] # type: ignore
    model.zero_grad()
    for xs, ys in zip(x.split(2), y.split(2)):
        (batch_loss(model, xs, ys) / 4).backward()
    for r, p in zip(reference, model.parameters()):
        assert torch.allclose(r, p.grad, atol=1e-6) # type: ignore
