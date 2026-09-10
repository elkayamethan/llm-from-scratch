import pytest

from data import ResumableSampler


def test_len_ignores_start_index() -> None:
    s = ResumableSampler(10, seed=1)
    s.set_epoch(0, start_index=7)
    assert len(s) == 10


def test_same_seed_and_epoch_same_permutation() -> None:
    a = ResumableSampler(100, seed=3)
    b = ResumableSampler(100, seed=3)
    a.set_epoch(2)
    b.set_epoch(2)
    assert list(a) == list(b)
    assert sorted(a) == list(range(100))


def test_epochs_differ() -> None:
    s = ResumableSampler(100, seed=3)
    s.set_epoch(0)
    e0 = list(s)
    s.set_epoch(1)
    assert list(s) != e0


def test_seeds_differ() -> None:
    a = ResumableSampler(100, seed=0)
    b = ResumableSampler(100, seed=1)
    a.set_epoch(1)
    b.set_epoch(0)
    assert list(a) != list(b)


def test_start_index_yields_tail() -> None:
    s = ResumableSampler(100, seed=3)
    s.set_epoch(4)
    full = list(s)
    s.set_epoch(4, start_index=37)
    assert list(s) == full[37:]


def test_set_epoch_resets_start_index() -> None:
    s = ResumableSampler(100, seed=3)
    s.set_epoch(0, start_index=50)
    s.set_epoch(0)
    assert len(list(s)) == 100


def test_iter_requires_set_epoch() -> None:
    with pytest.raises(ValueError, match="set_epoch"):
        list(ResumableSampler(10, seed=0))


@pytest.mark.parametrize("seed", [-1, 2**31])
def test_seed_bounds(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        ResumableSampler(10, seed=seed)


def test_start_index_bounds() -> None:
    s = ResumableSampler(10, seed=0)
    with pytest.raises(ValueError, match="start_index"):
        s.set_epoch(0, start_index=11)
