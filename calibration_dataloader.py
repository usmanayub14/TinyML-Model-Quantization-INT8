"""Deterministic synthetic ImageNet-shaped inputs without image downloads."""

import torch
from torch.utils.data import DataLoader, Dataset


INPUT_SHAPE = (3, 224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SyntheticCalibrationDataset(Dataset):
    """Generate each normalized RGB image from a private, per-index RNG.

    Samples are stable across repeated traversals and independent of global
    RNG state. Lazy generation avoids allocating the entire calibration set.
    Random pixels simulate the mechanics of calibration, not natural images.
    """

    def __init__(self, num_samples: int = 128, seed: int = 42) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        self.num_samples = num_samples
        self.seed = seed
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        pixels = torch.rand(INPUT_SHAPE, generator=generator)
        return (pixels - self.mean) / self.std


def get_calibration_dataloader(
    num_samples: int = 128, batch_size: int = 8, seed: int = 42
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        SyntheticCalibrationDataset(num_samples, seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
