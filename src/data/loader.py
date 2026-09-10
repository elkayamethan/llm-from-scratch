from torch import Tensor
from torch.utils.data import DataLoader
from data.sampler import ResumableSampler
from data.dataset import PretrainingDataset


def split_corpus(token_ids: Tensor, train_fraction: float) -> tuple[Tensor, Tensor]:
    """Returns (train_corpus, validation_corpus)"""
    if not (0 < train_fraction < 1):
        raise ValueError(f"train_fraction must be between 0 and 1, received: train_fraction={train_fraction}.")

    split_index = int(train_fraction * len(token_ids))
    return token_ids[:split_index], token_ids[split_index:]


def create_pretraining_dataloaders(
        train_dataset: PretrainingDataset,
        val_dataset: PretrainingDataset,
        *,
        seed: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_dataloader, val_dataloader).
    
    Note: train uses drop_last and a ResumableSampler.
    """

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=ResumableSampler(len(train_dataset), seed=seed)
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
