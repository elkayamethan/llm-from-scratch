import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data import PretrainingDataset


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, np.ndarray]:
    arr = np.random.default_rng(0).integers(0, 50257, size=10_007, dtype=np.uint16)
    path = tmp_path / "train.bin"
    arr.tofile(path)
    return path, arr


def test_matches_in_memory(corpus: tuple[Path, np.ndarray]) -> None:
    path, arr = corpus
    mm = PretrainingDataset.from_memmap(path, context_size=32, stride=16, dtype=np.uint16)
    ref = PretrainingDataset(torch.from_numpy(arr.astype(np.int64)), context_size=32, stride=16)
    assert len(mm) == len(ref)
    for i in (0, 1, len(ref) // 2, len(ref) - 1):
        x, y = mm[i]
        rx, ry = ref[i]
        assert x.dtype == torch.int64 and y.dtype == torch.int64
        assert torch.equal(x, rx) and torch.equal(y, ry)


def test_pickle_excludes_array(corpus: tuple[Path, np.ndarray]) -> None:
    path, _ = corpus
    ds = PretrainingDataset.from_memmap(path, context_size=32, stride=16, dtype=np.uint16)
    ds[0]
    assert len(pickle.dumps(ds)) < 2_000


def test_workers(corpus: tuple[Path, np.ndarray]) -> None:
    path, arr = corpus
    ds = PretrainingDataset.from_memmap(path, context_size=32, stride=32, dtype=np.uint16)
    x, y = next(iter(DataLoader(ds, batch_size=4, num_workers=2)))
    flat = torch.from_numpy(arr[: 4 * 32 + 1].astype(np.int64))
    assert torch.equal(x, flat[:-1].view(4, 32))
    assert torch.equal(y, flat[1:].view(4, 32))


def test_rejects_partial_item(tmp_path: Path) -> None:
    path = tmp_path / "odd.bin"
    path.write_bytes(b"\x00" * 7)
    with pytest.raises(ValueError, match="multiple of"):
        PretrainingDataset.from_memmap(path, context_size=2, stride=1, dtype=np.uint16)
