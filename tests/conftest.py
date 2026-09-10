import pytest

from common.config.schemas import GPTConfig, TrainingConfig


@pytest.fixture
def tiny_model_cfg() -> GPTConfig:
    return GPTConfig(
        context_length=16, embedding_dim=32, n_heads=4, n_transformer_layers=2,
        vocab_size=64, drop_rate=0.0, kqv_bias=True, attention_impl="manual",
    )


@pytest.fixture
def tiny_train_cfg() -> TrainingConfig:
    return TrainingConfig(
        n_epochs=2, batch_size=4, grad_accum_steps=2,
        max_lr=1e-3, min_lr=1e-4, warmup_steps=2, weight_decay=0.1, grad_clip_norm=1.0,
        betas=(0.9, 0.95), eps=1e-8, use_bf16=False, compile=False,
        eval_freq=10, eval_batches=2, checkpoint_freq=20, log_freq=4,
    )
