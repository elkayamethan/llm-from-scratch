from pathlib import Path

import torch
from torch.optim import AdamW

from architecture.models import GPT
from common.checkpoint import load_checkpoint, save_checkpoint
from common.config.schemas import GPTConfig, RunConfig, TrainingConfig, config_class
from training import param_groups
from training.loss import batch_loss


def _step(model: GPT, opt: AdamW, vocab: int) -> None:
    x = torch.randint(0, vocab, (2, 8))
    batch_loss(model, x, x).backward()
    opt.step()
    opt.zero_grad()


def test_roundtrip(tmp_path: Path, tiny_model_cfg: GPTConfig, tiny_train_cfg: TrainingConfig) -> None:
    torch.manual_seed(0)
    model = GPT(tiny_model_cfg)
    opt = AdamW(param_groups(model, 0.1), lr=1e-3, betas=(0.9, 0.95))
    _step(model, opt, tiny_model_cfg.vocab_size)
    run_cfg = RunConfig(run_name="t", seed=1, data_dir=tmp_path, checkpoint_dir=tmp_path,
                        model=tiny_model_cfg, training=tiny_train_cfg)
    history = {"lr": [1e-3] * 7, "grad_norm": [0.5] * 7}
    path = tmp_path / "ckpt" / "step_000007.pt"

    save_checkpoint(path, model=model, optimizer=opt, global_step=7, epoch=1,
                    run_cfg=run_cfg, history=history)
    assert path.is_file() and not path.with_name(path.name + ".tmp").exists()

    model2 = GPT(tiny_model_cfg)
    opt2 = AdamW(param_groups(model2, 0.1), lr=1e-3)
    state = load_checkpoint(path, model=model2, optimizer=opt2)

    for (name, a), (_, b) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(a, b), name
    assert state["global_step"] == 7 and state["epoch"] == 1 and state["history"] == history
    assert "generator_state" not in state
    assert "train_cfg" not in state
    assert "model_cfg" not in state
    assert state["run_cfg_class"] == "RunConfig"
    assert config_class(state["run_cfg_class"]).model_validate(state["run_cfg"]) == run_cfg
    assert opt2.param_groups[0]["betas"] == (0.9, 0.95)
