from __future__ import annotations

from collections.abc import Sequence


def get_train_transforms(
    patch_size: Sequence[int] | tuple[int, int, int] = (96, 96, 96)
) -> list[object]:
    try:
        from monai.transforms import (
            CropForegroundd,
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            RandCropByPosNegLabeld,
            RandFlipd,
            RandRotated,
            ScaleIntensityRanged,
            SpatialPadd,
        )

        preprocessing = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(
                keys=["image", "label"], source_key="image", margin=10
            ),  # crop out black borders - be careful with this
            ScaleIntensityRanged(keys=["image"], a_min=-250, a_max=600, b_min=0.0, b_max=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=patch_size,
                mode="constant",
            ),
        ]

        augments = [
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
            RandRotated(keys=["image", "label"], range_x=0.1, prob=0.2),
        ]

        return preprocessing + augments
    except Exception:  # pragma: no cover - optional dep
        return []


def get_val_transforms(
    patch_size: Sequence[int] | tuple[int, int, int] = (96, 96, 96)
) -> list[object]:
    try:
        from monai.transforms import (
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            RandCropByPosNegLabeld,
            ScaleIntensityRanged,
            SpatialPadd,
        )

        return [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=patch_size,
                mode="constant",
            ),
        ]
    except Exception:  # pragma: no cover - optional dep
        return []
