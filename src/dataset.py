"""CIFAR-10 dataset and DataLoader helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return the preprocessing pipeline for training or evaluation.

    Training uses random augmentation. Evaluation intentionally uses only
    deterministic preprocessing so that metrics are comparable between runs.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 if necessary and return train/test loaders.

    Args:
        data_dir: Directory used to cache the CIFAR-10 files.
        batch_size: Number of images yielded in each batch.
        num_workers: Number of worker processes used to load batches.
        pin_memory: Whether DataLoader workers pin tensors for faster CUDA copies.
            If omitted, it follows CUDA availability.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    loader_options = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options,
    )
    return train_loader, test_loader
