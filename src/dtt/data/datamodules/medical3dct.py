from __future__ import annotations

from typing import Any

from dtt.data.datamodules.custom_datasets.json_cache_ds import JSONCacheDataset
from dtt.utils.registry import register_datamodule


class _UnwrapDataset:
    """Wrapper dataset to unwrap list output from RandCropByPosNegLabeld.

    RandCropByPosNegLabeld returns a list even with num_samples=1,
    so we extract the first element to get the dictionary.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        result = self.dataset[idx]
        # If result is a list (from RandCropByPosNegLabeld), extract first element
        return result[0] if isinstance(result, list) else result


@register_datamodule("data.medical3dct")
def build_medical3dct_datamodule(cfg: dict[str, Any]):
    from lightning.pytorch import LightningDataModule

    data_cfg = cfg.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 2))
    params = data_cfg.get("params", {})

    json_train = params.get("json_train")
    json_val = params.get("json_val")
    cache_rate = float(params.get("cache_rate", 0.0))
    synthetic = bool(params.get("synthetic", True))
    spatial_size = tuple(params.get("spatial_size", [256, 256, 256]))
    pixdim = tuple(params.get("pixdim", [1.0, 1.0, 1.0]))
    random_patch = bool(params.get("random_patch", False))

    class _Synthetic3DDataset:
        def __init__(self, size: int = 64, length: int = 64):
            import torch

            self.size = size
            self.length = length
            self.rng = torch.Generator().manual_seed(0)

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, idx: int):
            import torch

            x = torch.randn(1, self.size, self.size, self.size, generator=self.rng)
            y = (x.mean() > 0).float().expand_as(x[:1])  # simple target
            # Return dictionary to match MONAI convention
            return {"image": x, "label": y}

    class Medical3DCTDataModule(LightningDataModule):
        def __init__(self) -> None:
            super().__init__()
            self._train = None
            self._val = None

        def setup(self, stage: str | None = None) -> None:  # type: ignore[override]
            if synthetic or not json_train or not json_val:
                # Fallback synthetic tiny dataset
                self._train = _Synthetic3DDataset(length=32)
                self._val = _Synthetic3DDataset(length=16)
            else:
                # JSON-based dataset using MONAI
                from dtt.data.transforms.medical3dct import (
                    get_train_transforms,
                    get_val_transforms,
                )

                train_ds = JSONCacheDataset(
                    data_dir=json_train,
                    cache_rate=cache_rate,
                    num_workers=num_workers,
                    transforms=get_train_transforms(
                        patch_size=spatial_size, pixdim=pixdim, random_patch=random_patch
                    ),
                )

                val_ds = JSONCacheDataset(
                    data_dir=json_val,
                    cache_rate=cache_rate,
                    num_workers=num_workers,
                    transforms=get_val_transforms(patch_size=spatial_size, pixdim=pixdim),
                )

                # Wrap datasets to handle list output from RandCropByPosNegLabeld
                self._train = _UnwrapDataset(train_ds)
                self._val = _UnwrapDataset(val_ds)

        def train_dataloader(self):  # type: ignore[override]
            from torch.utils.data import DataLoader

            return DataLoader(self._train, batch_size=batch_size, num_workers=num_workers)

        def val_dataloader(self):  # type: ignore[override]
            from torch.utils.data import DataLoader

            return DataLoader(self._val, batch_size=batch_size, num_workers=num_workers)

    return Medical3DCTDataModule()
