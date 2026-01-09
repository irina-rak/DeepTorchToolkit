from __future__ import annotations


def get_train_transforms(spatial_size: tuple[int, int] = (256, 256)):
    try:
        import numpy as np
        from monai.data import PILReader
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            LoadImaged,
            RandAffined,
            Resized,
            ScaleIntensityRanged,
        )

        # Use PILReader with mode='L' to force grayscale loading
        pil_reader = PILReader(converter=lambda img: img.convert("L"))

        # Note: allow_missing_keys=True allows datasets without labels (e.g., CXR8)
        preprocessing = [
            LoadImaged(keys=["image", "label"], reader=pil_reader, allow_missing_keys=True),
            EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
            ),
            Resized(
                keys=["image", "label"],
                spatial_size=spatial_size,
                mode=["bilinear", "nearest"],
                allow_missing_keys=True,
            ),
        ]

        augments = [
            RandAffined(
                keys=["image", "label"],
                rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                translate_range=[(-1, 1), (-1, 1)],
                scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                spatial_size=spatial_size,
                padding_mode="zeros",
                prob=0.5,
                allow_missing_keys=True,
            ),
        ]

        return Compose(preprocessing + augments)
    except Exception:  # pragma: no cover - optional dep
        return None


def get_val_transforms(spatial_size: tuple[int, int] = (256, 256)):
    try:
        from monai.data import PILReader
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            LoadImaged,
            Resized,
            ScaleIntensityRanged,
        )

        # Use PILReader with mode='L' to force grayscale loading
        pil_reader = PILReader(converter=lambda img: img.convert("L"))

        # Note: allow_missing_keys=True allows datasets without labels (e.g., CXR8)
        return Compose(
            [
                LoadImaged(keys=["image", "label"], reader=pil_reader, allow_missing_keys=True),
                EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
                ),
                Resized(
                    keys=["image", "label"],
                    spatial_size=spatial_size,
                    mode=["bilinear", "nearest"],
                    allow_missing_keys=True,
                ),
            ]
        )
    except Exception:  # pragma: no cover - optional dep
        return None
