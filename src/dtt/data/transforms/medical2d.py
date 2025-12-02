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

        preprocessing = [
            LoadImaged(keys=["image", "label"], reader=pil_reader),
            EnsureChannelFirstd(keys=["image", "label"]),
            # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),  # Explicitly scale to [0, 1]
            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
            ),
            Resized(
                keys=["image", "label"], spatial_size=spatial_size, mode=["bilinear", "nearest"]
            ),
            # ToTensord(keys=["image", "label"]),
        ]

        augments = [
            # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            # RandRotated(
            #     keys=["image", "label"], range_x=0.1, prob=0.2, mode=["bilinear", "nearest"]
            # ),
            RandAffined(
                keys=["image", "label"],
                rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                translate_range=[(-1, 1), (-1, 1)],
                scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                spatial_size=spatial_size,
                padding_mode="zeros",
                prob=0.5,
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

        return Compose(
            [
                LoadImaged(keys=["image", "label"], reader=pil_reader),
                EnsureChannelFirstd(keys=["image", "label"]),
                # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),  # Explicitly scale to [0, 1]
                ScaleIntensityRanged(
                    keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
                ),
                Resized(
                    keys=["image", "label"],
                    spatial_size=spatial_size,
                    mode=["bilinear", "nearest"],
                ),
            ]
        )
    except Exception:  # pragma: no cover - optional dep
        return None
