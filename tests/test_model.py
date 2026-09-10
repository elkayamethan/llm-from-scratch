import math

import torch

from architecture.models import GPT
from common.config.schemas import GPTConfig
from training.loss import batch_loss


def _cfg(d: int, layers: int, vocab: int) -> GPTConfig:
    return GPTConfig(context_length=64, embedding_dim=d, n_heads=4, n_transformer_layers=layers,
                     vocab_size=vocab, drop_rate=0.0, kqv_bias=True, attention_impl="manual")


def test_head_is_tied(tiny_model_cfg) -> None:
    m = GPT(tiny_model_cfg)
    assert m.out.weight is m.token_embedding.weight
    d, L, V, C = 32, 2, 64, 16
    expected = L * (12 * d * d + 13 * d) + V * d + C * d + 2 * d
    assert sum(p.numel() for p in m.parameters()) == expected


def test_init_statistics() -> None:
    torch.manual_seed(0)
    m = GPT(_cfg(d=256, layers=4, vocab=512))
    block = m.transformer_blocks[0]
    assert abs(block.ff[0].weight.std().item() - 0.02) < 0.003 # type: ignore
    assert abs(block.attention.W_q.weight.std().item() - 0.02) < 0.003 # type: ignore
    assert abs(m.token_embedding.weight.std().item() - 0.02) < 0.003
    assert abs(m.positional_embedding.weight.std().item() - 0.02) < 0.003
    scaled = 0.02 / math.sqrt(2 * 4)
    assert abs(block.ff[2].weight.std().item() - scaled) < 0.002 # type: ignore
    assert abs(block.attention.final_proj.weight.std().item() - scaled) < 0.002 # type: ignore
    assert torch.all(block.ff[0].bias == 0) and torch.all(block.attention.W_q.bias == 0) # type: ignore
    assert torch.all(m.final_norm.scale == 1) and torch.all(m.final_norm.shift == 0)


def test_initial_loss_near_uniform() -> None:
    torch.manual_seed(0)
    cfg = _cfg(d=128, layers=4, vocab=512)
    m = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 64))
    y = torch.randint(0, cfg.vocab_size, (4, 64))
    assert abs(batch_loss(m, x, y).item() - math.log(cfg.vocab_size)) < 0.5


def test_state_dict_roundtrip_keeps_tying(tiny_model_cfg) -> None:
    a = GPT(tiny_model_cfg)
    b = GPT(tiny_model_cfg)
    b.load_state_dict(a.state_dict())
    assert b.out.weight is b.token_embedding.weight
    assert torch.equal(b.out.weight, a.token_embedding.weight)
    assert {"out.weight", "token_embedding.weight"} <= set(a.state_dict())
