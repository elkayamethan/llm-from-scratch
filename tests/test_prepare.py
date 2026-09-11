import json
from pathlib import Path

import numpy as np

from cli.prepare_fineweb_edu import write_splits
from data.tokenizer import GPT2Tokenizer


def _docs() -> list[str]:
    return [f"Document number {i}. " + "word " * (i % 7 + 1) for i in range(40)]


def test_write_splits(tmp_path: Path) -> None:
    meta = write_splits(_docs(), tmp_path, val_tokens=50, num_workers=2, source="unit", shards=["a", "b"])

    tok = GPT2Tokenizer()
    expected = [np.asarray(tok.encode(d) + [tok.eot_id], dtype=np.uint16) for d in _docs()]
    val = np.fromfile(tmp_path / "val.bin", dtype=np.uint16)
    train = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)

    n_val = meta["val_docs"]
    assert np.array_equal(val, np.concatenate(expected[:n_val]))
    assert np.array_equal(train, np.concatenate(expected[n_val:]))
    assert len(val) >= 50 and len(np.concatenate(expected[: n_val - 1])) < 50
    assert val[-1] == tok.eot_id and train[-1] == tok.eot_id

    on_disk = json.loads((tmp_path / "meta.json").read_text())
    assert on_disk == meta
    assert meta["train_tokens"] == len(train) and meta["val_tokens"] == len(val)
    assert meta["train_docs"] + meta["val_docs"] == 40
    assert meta["tokenizer"] == "gpt2" and meta["vocab_size"] == 50257 and meta["eot_id"] == 50256
    assert meta["dtype"] == "uint16" and meta["source"] == "unit" and meta["shards"] == ["a", "b"]
