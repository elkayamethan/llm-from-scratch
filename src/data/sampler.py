from collections.abc import Iterator

import torch

from torch.utils.data import Sampler


class ResumableSampler(Sampler[int]):
    """
    Sampler that depends only on 'epoch' and 'seed', 'start_index' skips samples within-epoch.
    
    Note: __len__ ignores 'start_index'.
    """

    def __init__(self, num_samples: int, *, seed: int) -> None:
        super().__init__()

        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, received: {num_samples}")
        if seed < 0 or seed >= 2**31:
            raise ValueError(f"0 <= seed < 2^31 must hold, received: {seed}")

        self.num_samples = num_samples
        self.seed = seed
        self._epoch: int | None = None
        self._start_index = 0


    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        if 0 > epoch or epoch >= 2**32:
            raise ValueError(f"0 <= epoch < 2^32 must hold, received: {epoch}")
        if 0 > start_index or start_index > self.num_samples:
            raise ValueError(f"0 <= start_index <= self.num_samples must hold, received: {start_index}")

        self._epoch = epoch
        self._start_index = start_index


    def __iter__(self) -> Iterator[int]:
        if self._epoch is None:
            raise ValueError("set_epoch must be called before iterating")
    
        generator = torch.Generator().manual_seed((self.seed << 32) | self._epoch)
        perm = torch.randperm(self.num_samples, generator=generator)
        return iter(perm[self._start_index:].tolist())


    def __len__(self):
        return self.num_samples
