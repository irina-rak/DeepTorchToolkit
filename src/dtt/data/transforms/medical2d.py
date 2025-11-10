from __future__ import annotations


def get_train_transforms() -> list[object]:
    try:
        from monai.transforms import (
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            RandFlipd,
            RandRotated,
            ScaleIntensityRanged,
        )

        preprocessing = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]

        augments = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotated(keys=["image", "label"], range_x=0.1, prob=0.2),
        ]

        return preprocessing + augments
    except Exception:  # pragma: no cover - optional dep
        return []


def get_val_transforms() -> list[object]:
    try:
        from monai.transforms import (
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            ScaleIntensityRanged,
        )

        return [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]
    except Exception:  # pragma: no cover - optional dep
        return []
