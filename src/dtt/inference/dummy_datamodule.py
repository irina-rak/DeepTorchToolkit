"""Dummy DataModule for unconditional generation without test data."""

from __future__ import annotations

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """Dummy dataset that generates random noise for unconditional generation.

    This dataset provides random tensors with the specified shape, allowing
    models to generate samples without requiring actual test data.
    """

    def __init__(
        self,
        num_samples: int,
        spatial_dims: int,
        in_channels: int,
        spatial_size: list[int],
    ):
        """Initialize dummy dataset.

        Args:
            num_samples: Number of samples (batches * batch_size)
            spatial_dims: 2 or 3 for 2D/3D data
            in_channels: Number of input channels
            spatial_size: Spatial dimensions [H, W] for 2D or [D, H, W] for 3D
        """
        self.num_samples = num_samples
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.spatial_size = spatial_size

        # Build shape: [C, H, W] or [C, D, H, W]
        if spatial_dims == 2:
            self.shape = [in_channels] + spatial_size
        elif spatial_dims == 3:
            self.shape = [in_channels] + spatial_size
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a dummy batch item with random noise.

        Returns:
            Dict with 'image' key containing random tensor
        """
        # Return random noise as a placeholder
        # The actual noise for generation will be created in test_step
        return {
            "image": torch.randn(*self.shape),
        }


class DummyDataModule(LightningDataModule):
    """DataModule that provides dummy data for unconditional generation."""

    def __init__(
        self,
        num_batches: int = 10,
        batch_size: int = 16,
        spatial_dims: int = 2,
        in_channels: int = 1,
        spatial_size: list[int] | None = None,
    ):
        """Initialize dummy datamodule.

        Args:
            num_batches: Number of batches to generate
            batch_size: Batch size
            spatial_dims: 2 or 3 for 2D/3D data
            in_channels: Number of input channels
            spatial_size: Spatial dimensions [H, W] for 2D or [D, H, W] for 3D
        """
        super().__init__()
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.spatial_size = spatial_size or ([128, 128] if spatial_dims == 2 else [64, 128, 128])

        # Total number of samples
        self.num_samples = num_batches * batch_size

        # Initialize dataset immediately (not waiting for setup)
        self.test_dataset = DummyDataset(
            num_samples=self.num_samples,
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            spatial_size=self.spatial_size,
        )

    def setup(self, stage: str | None = None):
        """Setup datasets (dataset already created in __init__)."""
        pass

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader with dummy data."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # No need for multiple workers for dummy data
        )
