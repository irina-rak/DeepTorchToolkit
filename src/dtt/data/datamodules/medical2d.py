from __future__ import annotations

from typing import Any

from dtt.data.datamodules.custom_datasets.json_cache_ds import JSONCacheDataset
from dtt.utils.registry import register_datamodule


@register_datamodule("data.medical2d")
def build_medical2d_datamodule(cfg: dict[str, Any]):
    from lightning.pytorch import LightningDataModule

    data_cfg = cfg.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 2))
    params = data_cfg.get("params", {})

    json_train = params.get("json_train")
    json_val = params.get("json_val")
    json_test = params.get("json_test")
    cache_rate = float(params.get("cache_rate", 0.0))
    synthetic = bool(params.get("synthetic", True))
    spatial_size = tuple(params.get("spatial_size", [256, 256]))

    class _Synthetic2DDataset:
        def __init__(self, size: int = 64, length: int = 64):
            import torch

            self.size = size
            self.length = length
            self.rng = torch.Generator().manual_seed(0)

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, idx: int):
            import torch

            # Generate random noise from standard normal distribution
            x = torch.randn(1, self.size, self.size, generator=self.rng)

            # Normalize to [0, 1] range to match real image data
            # torch.randn produces ~N(0,1), so we clip to [-3, 3] (99.7% coverage)
            # then scale to [0, 1]
            x = torch.clamp(x, -3.0, 3.0)  # Clip outliers
            x = (x + 3.0) / 6.0  # Scale from [-3, 3] to [0, 1]

            y = (x.mean() > 0.5).float().expand_as(x[:1])  # simple target
            # Return dictionary to match MONAI convention
            return {"image": x, "label": y}

    class Medical2DDataModule(LightningDataModule):
        def __init__(self) -> None:
            super().__init__()
            self._train = None
            self._val = None
            self._test = None

        def setup(self, stage: str | None = None) -> None:  # type: ignore[override]
            if synthetic or not json_train or not json_val:
                # Fallback synthetic tiny dataset
                self._train = _Synthetic2DDataset(length=32)
                self._val = _Synthetic2DDataset(length=16)
                self._test = _Synthetic2DDataset(length=16)
            else:
                # JSON-based dataset using MONAI
                from dtt.data.transforms.medical2d import (
                    get_train_transforms,
                    get_val_transforms,
                )

                # Only load datasets needed for the current stage
                if stage in (None, "fit"):
                    self._train = JSONCacheDataset(
                        data_dir=json_train,
                        cache_rate=cache_rate,
                        num_workers=num_workers,
                        transforms=get_train_transforms(spatial_size=spatial_size),
                    )

                    self._val = JSONCacheDataset(
                        data_dir=json_val,
                        cache_rate=cache_rate,
                        num_workers=num_workers,
                        transforms=get_val_transforms(spatial_size=spatial_size),
                    )

                if stage in ("test", "predict"):
                    # Use test dataset if provided, otherwise fall back to val
                    test_json = json_test if json_test else json_val
                    self._test = JSONCacheDataset(
                        data_dir=test_json,
                        cache_rate=cache_rate,
                        num_workers=num_workers,
                        transforms=get_val_transforms(spatial_size=spatial_size),
                    )

        def train_dataloader(self):  # type: ignore[override]
            from torch.utils.data import DataLoader

            return DataLoader(self._train, batch_size=batch_size, num_workers=num_workers)

        def val_dataloader(self):  # type: ignore[override]
            from torch.utils.data import DataLoader

            return DataLoader(self._val, batch_size=batch_size, num_workers=num_workers)

        def test_dataloader(self):  # type: ignore[override]
            from torch.utils.data import DataLoader

            return DataLoader(self._test, batch_size=batch_size, num_workers=num_workers)

    return Medical2DDataModule()
