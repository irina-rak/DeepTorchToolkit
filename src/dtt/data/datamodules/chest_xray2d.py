from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from dtt.data.datamodules.custom_datasets.json_cache_ds import JSONCacheDataset
from dtt.utils.registry import register_datamodule


@register_datamodule("chest_xray2d")
def build_chest_xray2d_datamodule(cfg: Dict[str, Any]):
    from lightning.pytorch import LightningDataModule

    data_cfg = cfg.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 2))
    params = data_cfg.get("params", {})

    dir_train = params.get("dir_train", "")
    dir_val = params.get("dir_val", "")
    cache_rate = float(params.get("cache_rate", 0.0))
    num_workers = int(params.get("num_workers", 2))

    class Medical2DDataModule(LightningDataModule):
        def __init__(self) -> None:
            super().__init__()
            self._train = None
            self._val = None

        def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
            from dtt.data.transforms.medical2d import (
                get_train_transforms,
                get_val_transforms,
            )

            self._train = JSONCacheDataset(
                data_dir=dir_train,
                cache_rate=cache_rate,
                num_workers=num_workers,
                transforms=get_train_transforms(),
            )

            self._val = JSONCacheDataset(
                data_dir=dir_val,
                cache_rate=cache_rate,
                num_workers=num_workers,
                transforms=get_val_transforms(),
            )

        def train_dataloader(self):  # type: ignore[override]
            from torch.utils.data import DataLoader

            return DataLoader(self._train, batch_size=batch_size, num_workers=num_workers)

        def val_dataloader(self):  # type: ignore[override]
            from torch.utils.data import DataLoader

            return DataLoader(self._val, batch_size=batch_size, num_workers=num_workers)

    return Medical2DDataModule()
