import torch
import numpy as np

from numpy.typing import DTypeLike
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from data.tokenizer import Tokenizer


class _MemmapTokens:
    """Token file mapped lazily - once per process, on first access"""

    def __init__(self, path: Path, dtype: DTypeLike) -> None:
        self.path = path
        self.dtype = np.dtype(dtype)

        size = path.stat().st_size
        if size % self.dtype.itemsize != 0:
            raise ValueError(f"file size must be a multiple of the item size, received: {size} bytes at {path} and dtype.itemsize={{self.dtype.itemsize}}")

        self._length = size // self.dtype.itemsize
        self._array: np.memmap | None = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, s: slice) -> Tensor:
        if self._array is None:
            self._array = np.memmap(self.path, dtype=self.dtype, mode="c")

        return torch.from_numpy(self._array[s])

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_array": None}


class PretrainingDataset(Dataset):
    """Turns a contiguous corpus into input,target tensors using a sliding-window approach"""

    def __init__(
        self,
        token_ids: Tensor | _MemmapTokens,
        context_size: int,
        stride: int,
    ) -> None:
        super().__init__()

        if stride < 1:
            raise ValueError(f"Stride must be an int >= 1, received: stride={stride}.")
        if context_size <= 0:
            raise ValueError(f"Context size must be an int > 0, received: context_size={context_size}")
        if len(token_ids) < context_size + 1:
            raise ValueError(f"Corpus size must be at least 1 + context_size, received: len(token_ids)={len(token_ids)} , context_size: {context_size}.")

        self.token_ids = token_ids
        self.context_size = context_size
        self.stride = stride

        self.num_samples = ((len(self.token_ids) - (self.context_size + 1)) // self.stride) + 1


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.stride
        x = self.token_ids[start : self.context_size + start].long()
        y = self.token_ids[start + 1 : self.context_size + start + 1].long()

        return x,y


    def __len__(self) -> int:
        return self.num_samples


    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: Tokenizer,
        *,
        context_size: int,
        stride: int,
    ) -> "PretrainingDataset":
        """Tokenizes text and initializes PretrainingDataset"""
        
        token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        return cls(token_ids, context_size, stride)


    @classmethod
    def from_memmap(
        cls, 
        path: Path, 
        *, 
        context_size: int, 
        stride: int,
        dtype: DTypeLike = np.uint16,
    ) -> "PretrainingDataset":
        """Builds PretrainingDataset from a raw token file, loaded only on first access"""

        memmap = _MemmapTokens(path, dtype)
        return cls(memmap, context_size, stride)
