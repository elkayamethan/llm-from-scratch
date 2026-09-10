from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from common.config import load_config
from common.config.schemas import RunConfig, TrainingConfig, config_class


def _training_kwargs() -> dict:
    return dict(
        n_epochs=1, batch_size=16, max_lr=3e-4, min_lr=3e-5, warmup_steps=10,
        weight_decay=0.1, eval_freq=5, eval_batches=2, checkpoint_freq=10, log_freq=2,
    )


def test_training_defaults() -> None:
    cfg = TrainingConfig(**_training_kwargs())
    assert cfg.betas == (0.9, 0.95) and cfg.eps == 1e-8 and cfg.grad_accum_steps == 1


@pytest.mark.parametrize("betas", [(0.9, 1.0), (-0.1, 0.95)])
def test_betas_validated(betas: tuple[float, float]) -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(**_training_kwargs(), betas=betas)


def test_run_config_from_yaml(tmp_path: Path, tiny_model_cfg) -> None:
    doc = {
        "run_name": "smoke", "seed": 1, "data_dir": "datasets/smoke", "checkpoint_dir": "checkpoints",
        "model": tiny_model_cfg.model_dump(), "training": {**_training_kwargs(), "betas": [0.9, 0.95]},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(doc))
    cfg = load_config(path, RunConfig)
    assert cfg.model == tiny_model_cfg
    assert cfg.training.betas == (0.9, 0.95)
    assert isinstance(cfg.data_dir, Path) and cfg.resume_from is None and cfg.device is None
    assert cfg.num_workers == 0 and cfg.pin_memory is False and cfg.keep_last == 3


def test_run_config_rejects_unknown_key(tiny_model_cfg) -> None:
    with pytest.raises(ValidationError):
        RunConfig(run_name="x", seed=1, data_dir="d", checkpoint_dir="c", model=tiny_model_cfg,
                  training=_training_kwargs(), tokenizer="gpt2") # pyright: ignore[reportCallIssue]


def test_registry_has_run_config() -> None:
    assert config_class("RunConfig") is RunConfig
