"""Visualization utilities for evaluation of generative models.

This module provides visualization functions for comparing real and generated
image distributions, including:
- t-SNE/UMAP scatter plots of feature embeddings
- Side-by-side sample grids
- Pixel intensity histograms
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def plot_feature_tsne(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    output_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Create t-SNE scatter plot of real vs fake feature embeddings.

    Args:
        real_features: Feature tensor for real images (N, D).
        fake_features: Feature tensor for fake images (M, D).
        output_path: Path to save the plot.
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of t-SNE iterations.
        figsize: Figure size (width, height).
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("sklearn not available, skipping t-SNE plot")
        return

    # Combine features
    real_np = real_features.numpy()
    fake_np = fake_features.numpy()
    combined = np.vstack([real_np, fake_np])
    
    # Limit samples for speed if too many
    max_samples = 2000
    if len(combined) > max_samples:
        # Subsample proportionally
        n_real = min(len(real_np), max_samples // 2)
        n_fake = min(len(fake_np), max_samples // 2)
        real_idx = np.random.choice(len(real_np), n_real, replace=False)
        fake_idx = np.random.choice(len(fake_np), n_fake, replace=False)
        real_np = real_np[real_idx]
        fake_np = fake_np[fake_idx]
        combined = np.vstack([real_np, fake_np])
        logger.info(f"Subsampled to {len(combined)} samples for t-SNE")

    # Create labels
    labels = np.array(["Real"] * len(real_np) + ["Generated"] * len(fake_np))

    # Run t-SNE
    logger.info("Running t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(combined) - 1),
        max_iter=n_iter,
        random_state=42,
    )
    embeddings = tsne.fit_transform(combined)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Split embeddings back
    real_emb = embeddings[:len(real_np)]
    fake_emb = embeddings[len(real_np):]
    
    ax.scatter(
        real_emb[:, 0], real_emb[:, 1],
        c="#3498db", alpha=0.6, label=f"Real (n={len(real_np)})",
        s=30, edgecolors="none"
    )
    ax.scatter(
        fake_emb[:, 0], fake_emb[:, 1],
        c="#e74c3c", alpha=0.6, label=f"Generated (n={len(fake_np)})",
        s=30, edgecolors="none"
    )
    
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("Feature Space: Real vs Generated", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved t-SNE plot to {output_path}")


def _plot_sample_grid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    output_path: str,
    num_samples: int = 16,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Create side-by-side grid of real vs fake samples.

    Args:
        real_images: Real images tensor (N, C, H, W) or (N, C, D, H, W).
        fake_images: Fake images tensor.
        output_path: Path to save the plot.
        num_samples: Number of samples per side.
        figsize: Figure size (auto-calculated if None).
    """
    # Sample random indices
    n_real = min(num_samples, len(real_images))
    n_fake = min(num_samples, len(fake_images))
    
    real_idx = np.random.choice(len(real_images), n_real, replace=False)
    fake_idx = np.random.choice(len(fake_images), n_fake, replace=False)
    
    real_samples = real_images[real_idx]
    fake_samples = fake_images[fake_idx]
    
    # For 3D, take center slice
    if real_samples.dim() == 5:  # (N, C, D, H, W)
        center = real_samples.shape[2] // 2
        real_samples = real_samples[:, :, center, :, :]
        fake_samples = fake_samples[:, :, center, :, :]
    
    # Convert to numpy and squeeze channel if single channel
    real_np = real_samples.squeeze(1).numpy()
    fake_np = fake_samples.squeeze(1).numpy()
    
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(num_samples)))
    n_rows = int(np.ceil(num_samples / n_cols))
    
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 2)

    fig = plt.figure(figsize=figsize)
    
    # Outer grid: 1 row, 2 columns (Real vs Fake) with gap
    # width_ratios=[1, 1] means equal width, wspace=0.1 adds gap
    import matplotlib.gridspec as gridspec
    gs_outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
    
    # Inner grids (Real)
    gs_real = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, 
        subplot_spec=gs_outer[0], 
        wspace=0.05, hspace=0.05
    )
    
    # Inner grids (Fake)
    gs_fake = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, 
        subplot_spec=gs_outer[1], 
        wspace=0.05, hspace=0.05
    )
    
    # Add title for Real block (using invisible subplot)
    ax_real_bg = fig.add_subplot(gs_outer[0], frameon=False)
    ax_real_bg.set_title("Real", fontsize=18, fontweight="bold", color="#3498db", pad=20)
    ax_real_bg.set_xticks([])
    ax_real_bg.set_yticks([])
    
    # Add title for Fake block
    ax_fake_bg = fig.add_subplot(gs_outer[1], frameon=False)
    ax_fake_bg.set_title("Generated", fontsize=18, fontweight="bold", color="#e74c3c", pad=20)
    ax_fake_bg.set_xticks([])
    ax_fake_bg.set_yticks([])

    # Plot Real images
    for i in range(n_rows * n_cols):
        r = i // n_cols
        c = i % n_cols
        
        ax = fig.add_subplot(gs_real[r, c])
        if i < len(real_np):
            ax.imshow(real_np[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    # Plot Fake images
    for i in range(n_rows * n_cols):
        r = i // n_cols
        c = i % n_cols
        
        ax = fig.add_subplot(gs_fake[r, c])
        if i < len(fake_np):
            ax.imshow(fake_np[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    
    # output_path is absolute, so save straight to it
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved sample grid to {output_path}")


def plot_intensity_histogram(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    output_path: str,
    num_bins: int = 100,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Create overlaid histogram of pixel intensity distributions.

    Args:
        real_images: Real images tensor.
        fake_images: Fake images tensor.
        output_path: Path to save the plot.
        num_bins: Number of histogram bins.
        figsize: Figure size.
    """
    # Flatten all pixel values
    real_flat = real_images.flatten().numpy()
    fake_flat = fake_images.flatten().numpy()
    
    # Subsample if too many pixels
    max_pixels = 1_000_000
    if len(real_flat) > max_pixels:
        real_flat = np.random.choice(real_flat, max_pixels, replace=False)
    if len(fake_flat) > max_pixels:
        fake_flat = np.random.choice(fake_flat, max_pixels, replace=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(
        real_flat, bins=num_bins, alpha=0.6, density=True,
        label="Real", color="#3498db", edgecolor="none"
    )
    ax.hist(
        fake_flat, bins=num_bins, alpha=0.6, density=True,
        label="Generated", color="#e74c3c", edgecolor="none"
    )
    
    ax.set_xlabel("Pixel Intensity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pixel Intensity Distribution", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved intensity histogram to {output_path}")


def generate_evaluation_visualizations(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    output_dir: str,
    plot_tsne: bool = True,
    plot_sample_grid: bool = True,
    plot_histogram: bool = True,
    num_grid_samples: int = 16,
    seed: int = 42,
) -> dict[str, str]:
    """Generate all evaluation visualizations.

    Args:
        real_images: Real images tensor.
        fake_images: Fake/generated images tensor.
        real_features: Extracted features for real images.
        fake_features: Extracted features for fake images.
        output_dir: Directory to save plots.
        plot_tsne: Whether to generate t-SNE plot.
        plot_sample_grid: Whether to generate sample grid.
        plot_histogram: Whether to generate intensity histogram.
        num_grid_samples: Number of samples per side in grid.
        seed: Random seed for reproducible subsampling.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    if plot_tsne:
        tsne_path = str(output_dir / "tsne_features.png")
        plot_feature_tsne(real_features, fake_features, tsne_path)
        saved_plots["tsne"] = tsne_path
    
    if plot_sample_grid:
        grid_path = str(output_dir / "sample_grid.png")
        _plot_sample_grid(real_images, fake_images, grid_path, num_samples=num_grid_samples)
        saved_plots["sample_grid"] = grid_path
    
    if plot_histogram:
        hist_path = str(output_dir / "intensity_histogram.png")
        plot_intensity_histogram(real_images, fake_images, hist_path)
        saved_plots["histogram"] = hist_path
    
    return saved_plots
