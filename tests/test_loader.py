import torch

from data import PretrainingDataset, ResumableSampler
from data import create_pretraining_dataloaders


def _datasets() -> tuple[PretrainingDataset, PretrainingDataset]:
    g = torch.Generator().manual_seed(0)
    train = PretrainingDataset(torch.randint(0, 64, (500,), generator=g), context_size=8, stride=8)
    val = PretrainingDataset(torch.randint(0, 64, (100,), generator=g), context_size=8, stride=8)
    return train, val


def test_loader_configuration() -> None:
    train_ds, val_ds = _datasets()
    train, val = create_pretraining_dataloaders(train_ds, val_ds, batch_size=4, seed=7)
    assert isinstance(train.sampler, ResumableSampler)
    assert train.batch_size == 4 and val.batch_size == 4
    assert train.drop_last and not val.drop_last
    assert len(train) == len(train_ds) // 4


def test_val_loader_is_sequential() -> None:
    train_ds, val_ds = _datasets()
    _, val = create_pretraining_dataloaders(train_ds, val_ds, batch_size=4, seed=7)
    x, _ = next(iter(val))
    assert torch.equal(x, torch.stack([val_ds[i][0] for i in range(4)]))


def test_train_batches_follow_sampler() -> None:
    train_ds, val_ds = _datasets()
    train, _ = create_pretraining_dataloaders(train_ds, val_ds, batch_size=4, seed=7)
    sampler = train.sampler
    assert isinstance(sampler, ResumableSampler)
    sampler.set_epoch(0)
    order = list(train.sampler)
    it = iter(train)
    for b in range(2):
        x, _ = next(it)
        assert torch.equal(x, torch.stack([train_ds[i][0] for i in order[4 * b: 4 * b + 4]]))
