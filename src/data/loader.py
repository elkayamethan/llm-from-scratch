import torch

from torch import Tensor
from torch.utils.data import DataLoader
from data.dataset import PretrainingDataset


def split_corpus(token_ids: Tensor, train_fraction: float) -> tuple[Tensor, Tensor]:
    """Returns (train_corpus, validation_corpus)"""
    if not (0 < train_fraction < 1):
        raise ValueError(f"train_fraction must be between 0 and 1, received: train_fraction={train_fraction}.")

    split_index = int(train_fraction * len(token_ids))
    return token_ids[:split_index], token_ids[split_index:]


def create_pretraining_dataloaders(
        token_ids: Tensor,
        train_fraction: float,
        *,
        context_size: int,
        train_stride: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Splits corpus and returns (training_dataloader, validation_dataloader)"""

    train_token_ids, val_token_ids = split_corpus(token_ids, train_fraction)

    train_dataset = PretrainingDataset(
        token_ids=train_token_ids,
        context_size=context_size,
        stride=train_stride,
    )
    val_dataset = PretrainingDataset(
        token_ids=val_token_ids,
        context_size=context_size,
        stride=context_size,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, val_dataloader
