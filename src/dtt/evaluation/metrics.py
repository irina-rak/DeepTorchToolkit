"""FID and KID metric computation.

This module provides functions for computing distribution-based metrics
between sets of image features:

- FID (Fréchet Inception Distance): Measures the distance between two
  Gaussian distributions fitted to the feature representations.
- KID (Kernel Inception Distance): Uses Maximum Mean Discrepancy (MMD)
  with a polynomial kernel. More robust for smaller sample sizes.

These metrics are commonly used to evaluate the quality of generative models
by comparing the distribution of generated images to real images.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def compute_fid(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
) -> float:
    """Compute Fréchet Inception Distance between two feature sets.

    FID measures the distance between two multivariate Gaussian distributions
    fitted to the feature representations of real and generated images.

    Lower FID indicates better quality (more similar distributions).

    Args:
        real_features: Features from real images, shape (N, D)
        fake_features: Features from generated images, shape (M, D)

    Returns:
        FID score (float). Lower is better.

    Note:
        Requires at least 2 samples in each set to compute statistics.
        For reliable FID scores, use at least 2048 samples.
    """
    # Convert to numpy for scipy operations
    real_features = real_features.cpu().numpy().astype(np.float64)
    fake_features = fake_features.cpu().numpy().astype(np.float64)

    # Compute mean and covariance for real features
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    # Compute mean and covariance for fake features
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Compute FID
    fid = _calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return float(fid)


def _calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculate Fréchet Distance between two Gaussian distributions.

    The Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:

        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

    Args:
        mu1: Mean of the first Gaussian.
        sigma1: Covariance matrix of the first Gaussian.
        mu2: Mean of the second Gaussian.
        sigma2: Covariance matrix of the second Gaussian.
        eps: Small constant for numerical stability.

    Returns:
        The Fréchet distance.

    Raises:
        ValueError: If the covariance matrices contain imaginary components.
    """
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different shapes"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different shapes"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        logger.warning(
            "FID calculation: singular product; adding epsilon to diagonal of covariances"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    # Numerical precision can cause tiny negative values, clamp to 0
    return max(0.0, fid)


def compute_kid(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    num_subsets: int = 100,
    subset_size: int = 1000,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> tuple[float, float]:
    """Compute Kernel Inception Distance between two feature sets.

    KID uses Maximum Mean Discrepancy (MMD) with a polynomial kernel to
    measure the distance between feature sets. Unlike FID, it doesn't
    assume Gaussian distributions, making it more robust for smaller samples.

    Lower KID indicates better quality (more similar distributions).

    Args:
        real_features: Features from real images, shape (N, D)
        fake_features: Features from generated images, shape (M, D)
        num_subsets: Number of random subsets to compute MMD over.
        subset_size: Size of each subset. If larger than available samples,
            uses all available samples.
        degree: Degree of the polynomial kernel.
        gamma: Kernel coefficient. If None, uses 1/feature_dim.
        coef0: Independent term in kernel function.

    Returns:
        Tuple of (KID mean, KID std) computed over subsets.

    Note:
        Unlike FID, KID is unbiased and works well with smaller sample sizes.
    """
    # Convert to numpy
    real_features = real_features.cpu().numpy().astype(np.float64)
    fake_features = fake_features.cpu().numpy().astype(np.float64)

    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    feature_dim = real_features.shape[1]

    # Adjust subset size if needed
    max_subset_size = min(n_real, n_fake)
    if subset_size > max_subset_size:
        subset_size = max_subset_size
        logger.info(f"Adjusted subset_size to {subset_size} due to sample size")

    if gamma is None:
        gamma = 1.0 / feature_dim

    # Compute KID over multiple subsets
    kid_values = []

    for _ in range(num_subsets):
        # Random subset selection
        real_idx = np.random.choice(n_real, subset_size, replace=False)
        fake_idx = np.random.choice(n_fake, subset_size, replace=False)

        real_subset = real_features[real_idx]
        fake_subset = fake_features[fake_idx]

        # Compute MMD
        mmd = _compute_mmd(
            real_subset, fake_subset,
            degree=degree, gamma=gamma, coef0=coef0
        )
        kid_values.append(mmd)

    kid_mean = float(np.mean(kid_values))
    kid_std = float(np.std(kid_values))

    return kid_mean, kid_std


def _compute_mmd(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 1.0,
) -> float:
    """Compute Maximum Mean Discrepancy with polynomial kernel.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    where k is the polynomial kernel: k(a,b) = (gamma * a·b + coef0)^degree

    Args:
        x: First set of samples, shape (N, D)
        y: Second set of samples, shape (M, D)
        degree: Polynomial degree.
        gamma: Scaling factor.
        coef0: Constant term.

    Returns:
        Unbiased MMD^2 estimate.
    """
    n = x.shape[0]
    m = y.shape[0]

    # Compute Gram matrices
    # k(a, b) = (gamma * a·b + coef0)^degree
    k_xx = (gamma * x @ x.T + coef0) ** degree
    k_yy = (gamma * y @ y.T + coef0) ** degree
    k_xy = (gamma * x @ y.T + coef0) ** degree

    # Unbiased MMD^2 estimator
    # E[k(x,x')] for x != x' (exclude diagonal)
    sum_xx = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
    sum_yy = (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
    sum_xy = np.sum(k_xy) / (n * m)

    mmd2 = sum_xx + sum_yy - 2 * sum_xy

    return mmd2


def compute_distribution_metrics(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    compute_kid_metric: bool = True,
    kid_num_subsets: int = 100,
    kid_subset_size: int = 1000,
) -> dict[str, float]:
    """Compute all distribution-based metrics.

    A convenience function that computes both FID and optionally KID.

    Args:
        real_features: Features from real images, shape (N, D)
        fake_features: Features from generated images, shape (M, D)
        compute_kid_metric: Whether to also compute KID.
        kid_num_subsets: Number of subsets for KID computation.
        kid_subset_size: Size of each subset for KID.

    Returns:
        Dictionary containing:
            - "fid": Fréchet Inception Distance
            - "kid_mean": KID mean (if compute_kid_metric=True)
            - "kid_std": KID standard deviation (if compute_kid_metric=True)
            - "n_real": Number of real samples
            - "n_fake": Number of fake samples
    """
    results = {
        "n_real": real_features.shape[0],
        "n_fake": fake_features.shape[0],
    }

    # Compute FID
    results["fid"] = compute_fid(real_features, fake_features)

    # Optionally compute KID
    if compute_kid_metric:
        kid_mean, kid_std = compute_kid(
            real_features, fake_features,
            num_subsets=kid_num_subsets,
            subset_size=kid_subset_size,
        )
        results["kid_mean"] = kid_mean
        results["kid_std"] = kid_std

    return results
