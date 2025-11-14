from __future__ import annotations

from collections.abc import Sequence


def get_train_transforms(
    patch_size: Sequence[int] | tuple[int, int, int] = (96, 96, 96),
    pixdim: Sequence[float] | tuple[float, float, float] = (1.0, 1.0, 1.0),
    random_patch: bool = False,
) -> list[object]:
    try:
        from monai.transforms import (
            CenterSpatialCropd,
            Compose,
            CropForegroundd,
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            Orientationd,
            RandCropByPosNegLabeld,
            RandFlipd,
            RandRotated,
            ScaleIntensityRanged,
            Spacingd,
            SpatialPadd,
        )

        preprocessing = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear")),
            CropForegroundd(
                keys=["image", "label"], source_key="image", margin=10
            ),  # crop out black borders - be careful with this
            ScaleIntensityRanged(keys=["image"], a_min=-250, a_max=600, b_min=0.0, b_max=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]

        if random_patch:
            preprocessing += [
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
            ]
        else:
            preprocessing += [
                CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=patch_size,
                ),
            ]

        preprocessing += [
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

        # Note: RandCropByPosNegLabeld returns a list even with num_samples=1
        # The list is unwrapped at the dataset level (see _UnwrapDataset in medical3dct.py)
        return Compose(preprocessing + augments)
    except Exception:  # pragma: no cover - optional dep
        return None


def get_val_transforms(
    patch_size: Sequence[int] | tuple[int, int, int] = (96, 96, 96),
    pixdim: Sequence[float] | tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[object]:
    try:
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            Orientationd,
            RandCropByPosNegLabeld,
            ScaleIntensityRanged,
            Spacingd,
            SpatialPadd,
        )

        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear")),
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
        )
        # Note: RandCropByPosNegLabeld returns a list even with num_samples=1
        # The list is unwrapped at the dataset level (see _UnwrapDataset in medical3dct.py)
    except Exception:  # pragma: no cover - optional dep
        return None
