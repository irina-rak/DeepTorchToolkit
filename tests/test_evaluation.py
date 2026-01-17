"""Tests for the evaluation module.

These tests verify the evaluation module's core functionality:
- Feature extractors initialization and output dimensions
- FID/KID metric computation with synthetic data
- Image loading utilities
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


class TestMetrics:
    """Test FID and KID metric computation."""

    def test_compute_fid_identical_distributions(self):
        """FID should be close to 0 for identical distributions."""
        from dtt.evaluation.metrics import compute_fid

        # Same distribution should have FID close to 0
        features = torch.randn(100, 64)
        fid = compute_fid(features, features.clone())

        assert fid >= 0, "FID should be non-negative"
        # With same data, FID should be very small
        assert fid < 1e-5, f"FID should be ~0 for identical distributions, got {fid}"

    def test_compute_fid_different_distributions(self):
        """FID should be positive for different distributions."""
        from dtt.evaluation.metrics import compute_fid

        # Different distributions should have positive FID
        real_features = torch.randn(100, 64)
        # Shift the mean significantly
        fake_features = torch.randn(100, 64) + 5.0

        fid = compute_fid(real_features, fake_features)

        assert fid > 0, "FID should be positive for different distributions"

    def test_compute_kid_identical_distributions(self):
        """KID should be close to 0 for identical distributions."""
        from dtt.evaluation.metrics import compute_kid

        features = torch.randn(200, 64)
        kid_mean, kid_std = compute_kid(features, features.clone(), num_subsets=10, subset_size=50)

        # KID mean should be small for identical distributions
        assert abs(kid_mean) < 0.1, f"KID mean should be ~0, got {kid_mean}"

    def test_compute_kid_different_distributions(self):
        """KID should be positive for different distributions."""
        from dtt.evaluation.metrics import compute_kid

        real_features = torch.randn(200, 64)
        fake_features = torch.randn(200, 64) + 3.0

        kid_mean, kid_std = compute_kid(
            real_features, fake_features, num_subsets=10, subset_size=50
        )

        assert kid_mean > 0, "KID should be positive for different distributions"
        assert kid_std >= 0, "KID std should be non-negative"

    def test_compute_distribution_metrics(self):
        """Test the convenience function for computing all metrics."""
        from dtt.evaluation.metrics import compute_distribution_metrics

        real_features = torch.randn(100, 64)
        fake_features = torch.randn(100, 64)

        results = compute_distribution_metrics(
            real_features,
            fake_features,
            compute_kid_metric=True,
            kid_num_subsets=5,
            kid_subset_size=50,
        )

        assert "fid" in results
        assert "kid_mean" in results
        assert "kid_std" in results
        assert "n_real" in results
        assert "n_fake" in results
        assert results["n_real"] == 100
        assert results["n_fake"] == 100


class TestFeatureExtractors:
    """Test feature extractors initialization and forward pass."""

    @pytest.mark.heavy
    def test_feature_extractor_2d_init(self):
        """Test 2D feature extractor initialization."""
        from dtt.evaluation.feature_extractors import FeatureExtractor2D

        extractor = FeatureExtractor2D()
        assert extractor.output_dim == 2048

    @pytest.mark.heavy
    def test_feature_extractor_2d_forward(self):
        """Test 2D feature extractor forward pass."""
        from dtt.evaluation.feature_extractors import FeatureExtractor2D

        extractor = FeatureExtractor2D()
        extractor.eval()

        # Test with grayscale input
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            features = extractor(x)

        assert features.shape == (2, 2048)

    @pytest.mark.heavy
    def test_feature_extractor_2d_rgb(self):
        """Test 2D feature extractor with RGB input."""
        from dtt.evaluation.feature_extractors import FeatureExtractor2D

        extractor = FeatureExtractor2D()
        extractor.eval()

        # Test with RGB input
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            features = extractor(x)

        assert features.shape == (2, 2048)

    @pytest.mark.heavy
    def test_get_feature_extractor_auto_2d(self):
        """Test factory function for 2D."""
        from dtt.evaluation.feature_extractors import get_feature_extractor

        extractor = get_feature_extractor(spatial_dims=2)
        assert extractor.output_dim == 2048

    @pytest.mark.heavy
    def test_get_feature_extractor_auto_3d(self):
        """Test factory function for 3D (may fail if MONAI not installed)."""
        pytest.importorskip("monai")
        from dtt.evaluation.feature_extractors import get_feature_extractor

        # Use smaller model for faster testing
        extractor = get_feature_extractor(
            spatial_dims=3, model_depth=10, pretrained=False  # Skip downloading weights for testing
        )
        assert extractor.output_dim == 512


class TestImageLoading:
    """Test image loading utilities."""

    def test_load_images_2d_png(self):
        """Test loading 2D PNG images."""
        from PIL import Image

        from dtt.evaluation.evaluate import load_images_from_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8))
                img.save(Path(tmpdir) / f"test_{i}.png")

            images = load_images_from_directory(tmpdir, spatial_dims=2)

            assert images.shape[0] == 5
            assert images.shape[1] == 1  # Grayscale
            assert images.shape[2] == 64
            assert images.shape[3] == 64
            assert images.min() >= 0 and images.max() <= 1

    def test_load_images_3d_npy(self):
        """Test loading 3D NPY volumes."""
        from dtt.evaluation.evaluate import load_images_from_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test volumes
            for i in range(3):
                vol = np.random.randn(32, 64, 64).astype(np.float32)
                np.save(Path(tmpdir) / f"volume_{i}.npy", vol)

            images = load_images_from_directory(tmpdir, spatial_dims=3)

            assert images.shape[0] == 3
            assert images.shape[1] == 1  # Single channel
            assert images.shape[2] == 32
            assert images.shape[3] == 64
            assert images.shape[4] == 64

    def test_load_images_max_samples(self):
        """Test max_samples parameter."""
        from PIL import Image

        from dtt.evaluation.evaluate import load_images_from_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 10 test images
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 256, (32, 32), dtype=np.uint8))
                img.save(Path(tmpdir) / f"test_{i}.png")

            # Load only 5
            images = load_images_from_directory(tmpdir, spatial_dims=2, max_samples=5)

            assert images.shape[0] == 5
