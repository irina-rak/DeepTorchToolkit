"""Evaluation module for generative models.

This module provides tools for evaluating the quality of generated images
using distribution-based metrics like FID (Fréchet Inception Distance)
and KID (Kernel Inception Distance).

Main components:
- Feature extractors for 2D (InceptionV3) and 3D (MedicalNet) images
- FID/KID computation with support for custom feature extractors
- CLI integration for easy evaluation of generated images

Example usage:
    from dtt.evaluation import evaluate_generated_images

    results = evaluate_generated_images(
        real_dir="/path/to/real/images",
        fake_dir="/path/to/generated/images",
        spatial_dims=2,
    )
    print(f"FID: {results['fid']:.2f}")
    print(f"KID: {results['kid_mean']:.4f} ± {results['kid_std']:.4f}")
"""

from dtt.evaluation.evaluate import evaluate_generated_images
from dtt.evaluation.feature_extractors import (
    FeatureExtractor2D,
    FeatureExtractor3D,
    get_feature_extractor,
)
from dtt.evaluation.metrics import compute_fid, compute_kid

__all__ = [
    "evaluate_generated_images",
    "FeatureExtractor2D",
    "FeatureExtractor3D",
    "get_feature_extractor",
    "compute_fid",
    "compute_kid",
]
