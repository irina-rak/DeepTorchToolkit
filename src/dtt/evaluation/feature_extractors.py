"""Feature extractors for evaluation metrics.

This module provides feature extractors for computing embeddings used in
FID and KID calculations. It supports both 2D and 3D medical images.

Feature Extractors:
- FeatureExtractor2D: InceptionV3 pretrained on ImageNet (2048-dim features)
- FeatureExtractor3D: MedicalNet 3D-ResNet pretrained on medical images
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureExtractor2D(nn.Module):
    """InceptionV3-based feature extractor for 2D images.

    Uses ImageNet-pretrained InceptionV3 weights (pool3 layer, 2048-dim features).
    Handles grayscale→RGB conversion for single-channel medical images.

    The Inception network requires input images of size 299x299. This extractor
    automatically resizes images if needed.

    Attributes:
        feature_dim: Output feature dimension (2048 for InceptionV3)
    """

    feature_dim: int = 2048

    def __init__(self, normalize_input: bool = True):
        """Initialize the 2D feature extractor.

        Args:
            normalize_input: If True, expects inputs in [0, 1] and normalizes
                them to ImageNet statistics. If False, expects pre-normalized inputs.
        """
        super().__init__()
        self.normalize_input = normalize_input

        # Lazy import to avoid loading torchvision if not needed
        try:
            from torchvision.models import Inception_V3_Weights, inception_v3
        except ImportError as e:
            raise ImportError(
                "torchvision is required for 2D feature extraction. "
                "Install it with: pip install torchvision"
            ) from e

        # Load InceptionV3 with pretrained weights
        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,  # We handle normalization ourselves
        )

        # Remove the classification head - we want features, not predictions
        self.inception.fc = nn.Identity()

        # Set to eval mode and freeze parameters
        self.inception.eval()
        for param in self.inception.parameters():
            param.requires_grad = False

        # ImageNet normalization statistics
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W) where C is 1 or 3.
               Values should be in [0, 1] if normalize_input=True.

        Returns:
            Feature tensor of shape (B, 2048)
        """
        # Handle grayscale images by repeating to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(
                f"Expected 1 or 3 channels, got {x.shape[1]}. "
                "For multi-channel images, consider averaging or selecting channels."
            )

        # Resize to 299x299 if needed (InceptionV3 input size)
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # Normalize to ImageNet statistics
        if self.normalize_input:
            x = (x - self.mean) / self.std

        # Extract features
        with torch.no_grad():
            features = self.inception(x)

        return features

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim


class FeatureExtractor3D(nn.Module):
    """MedicalNet 3D-ResNet feature extractor for 3D medical volumes.

    Uses 3D-ResNet pretrained on 23 medical imaging datasets from the
    MedicalNet project (https://github.com/Tencent/MedicalNet).

    This provides domain-appropriate features for 3D medical images,
    as opposed to using 2D extractors on individual slices.

    Attributes:
        feature_dim: Output feature dimension (depends on model depth)
    """

    # Feature dimensions for different model depths
    _feature_dims = {
        10: 512,
        18: 512,
        34: 512,
        50: 2048,
    }

    def __init__(
        self,
        model_depth: Literal[10, 18, 34, 50] = 50,
        pretrained: bool = True,
        normalize_input: bool = True,
    ):
        """Initialize the 3D feature extractor.

        Args:
            model_depth: ResNet depth (10, 18, 34, or 50). Default 50.
            pretrained: Whether to load MedicalNet pretrained weights.
            normalize_input: If True, normalizes inputs to zero mean and unit variance.
        """
        super().__init__()
        self.model_depth = model_depth
        self.normalize_input = normalize_input
        self.feature_dim = self._feature_dims[model_depth]

        # Build the 3D ResNet model
        self.model = self._build_resnet3d(model_depth)

        if pretrained:
            self._load_pretrained_weights(model_depth)

        # Set to eval mode and freeze parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_resnet3d(self, depth: int) -> nn.Module:
        """Build a 3D ResNet model architecture.

        This is a simplified implementation of 3D ResNet that matches
        the MedicalNet architecture for loading pretrained weights.
        """
        try:
            from monai.networks.nets import ResNet
        except ImportError as e:
            raise ImportError(
                "MONAI is required for 3D feature extraction. "
                "Install it with: pip install monai"
            ) from e

        # Map depth to MONAI's block configuration
        # MONAI's ResNet uses similar architecture to MedicalNet
        depth_to_blocks = {
            10: (1, 1, 1, 1),
            18: (2, 2, 2, 2),
            34: (3, 4, 6, 3),
            50: (3, 4, 6, 3),
        }

        block_type = "bottleneck" if depth >= 50 else "basic"
        blocks = depth_to_blocks[depth]

        model = ResNet(
            block=block_type,
            layers=blocks,
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,  # Will be replaced with identity
        )

        # Remove the classification head
        model.fc = nn.Identity()

        return model

    def _load_pretrained_weights(self, depth: int) -> None:
        """Load MedicalNet pretrained weights.

        Weights are downloaded from the MedicalNet repository on first use.
        """
        import os
        from pathlib import Path

        # Check for cached weights
        cache_dir = Path.home() / ".cache" / "medicalnet"
        cache_dir.mkdir(parents=True, exist_ok=True)

        weight_file = cache_dir / f"resnet_{depth}_23dataset.pth"

        if not weight_file.exists():
            logger.info(f"Downloading MedicalNet ResNet-{depth} weights...")
            try:
                # Try to download from the official repository
                url = f"https://github.com/Tencent/MedicalNet/raw/master/pretrain/resnet_{depth}_23dataset.pth"
                torch.hub.download_url_to_file(url, str(weight_file))
                logger.info(f"Downloaded weights to {weight_file}")
            except Exception as e:
                logger.warning(
                    f"Could not download MedicalNet weights: {e}. "
                    "Using random initialization. For best results, manually download "
                    f"weights from https://github.com/Tencent/MedicalNet"
                )
                return

        # Load weights
        try:
            state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
            # MedicalNet saves with 'state_dict' key
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Remove 'module.' prefix if present (from DataParallel training)
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }

            # Remove classification head weights
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("fc.") and not k.startswith("conv_seg.")
            }

            # Load with strict=False to handle minor architecture differences
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded MedicalNet ResNet-{depth} pretrained weights")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input 3D volumes.

        Args:
            x: Input tensor of shape (B, C, D, H, W) where C is typically 1.
               Values should be in [0, 1] if normalize_input=True.

        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Handle multi-channel inputs by averaging
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        # Normalize input
        if self.normalize_input:
            # Per-sample normalization to zero mean and unit variance
            mean = x.mean(dim=(2, 3, 4), keepdim=True)
            std = x.std(dim=(2, 3, 4), keepdim=True) + 1e-8
            x = (x - mean) / std

        # Extract features
        with torch.no_grad():
            features = self.model(x)

        # Ensure output is 2D (B, features)
        if features.dim() > 2:
            features = F.adaptive_avg_pool3d(features, 1).flatten(1)

        return features

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim


def get_feature_extractor(
    spatial_dims: int = 2,
    extractor_type: str = "auto",
    **kwargs,
) -> nn.Module:
    """Factory function to get the appropriate feature extractor.

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        extractor_type: Type of extractor to use:
            - "auto": Automatically select based on spatial_dims
            - "inception": InceptionV3 for 2D (forces 2D even for 3D data)
            - "medicalnet": MedicalNet 3D-ResNet (only for 3D data)
        **kwargs: Additional arguments passed to the extractor constructor.

    Returns:
        Feature extractor module.

    Raises:
        ValueError: If invalid combination of spatial_dims and extractor_type.
    """
    if extractor_type == "auto":
        extractor_type = "inception" if spatial_dims == 2 else "medicalnet"

    if extractor_type == "inception":
        return FeatureExtractor2D(**kwargs)
    elif extractor_type == "medicalnet":
        if spatial_dims != 3:
            raise ValueError("MedicalNet extractor requires spatial_dims=3")
        return FeatureExtractor3D(**kwargs)
    else:
        raise ValueError(
            f"Unknown extractor type: {extractor_type}. "
            "Choose from: 'auto', 'inception', 'medicalnet'"
        )
