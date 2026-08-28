import torch

from torch.utils.data import Dataset
from torch import Tensor
from data.tokenizer import Tokenizer


class PretrainingDataset(Dataset):
    """Turns a contiguous corpus into input,target tensors using a sliding-window approach"""

    def __init__(
        self,
        token_ids: Tensor,
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
        context_size: int,
        stride: int,
    ) -> "PretrainingDataset":
        """Tokenizes text and initializes PretrainingDataset"""
        
        token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        return cls(token_ids, context_size, stride)
