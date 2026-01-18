"""Paired metrics for conditional generation evaluation.

This module computes per-image metrics when generated images can be
paired with their corresponding source images (e.g., in conditional generation).

Metrics:
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
    spatial_dims: int = 2,
) -> float:
    """Compute SSIM between two images.

    Args:
        img1: First image tensor (C, H, W) or (C, D, H, W).
        img2: Second image tensor.
        data_range: Data range (1.0 for [0,1], 255 for [0,255]).
        spatial_dims: 2 for 2D images, 3 for 3D volumes.

    Returns:
        SSIM value (0 to 1, higher = more similar).
    """
    try:
        from monai.metrics import SSIMMetric
    except ImportError:
        raise ImportError("MONAI is required for SSIM. Install with: pip install monai")

    # Add batch dimension
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    ssim_metric = SSIMMetric(spatial_dims=spatial_dims, data_range=data_range)
    ssim_val = ssim_metric(img1, img2)
    
    return ssim_val.item()


def compute_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    net: str = "alex",
    device: str = "cuda",
) -> float:
    """Compute LPIPS between two images.

    LPIPS uses deep features from pretrained networks to measure
    perceptual similarity. Lower = more similar.

    Args:
        img1: First image tensor (C, H, W), should be normalized to [-1, 1].
        img2: Second image tensor.
        net: Network to use ('alex', 'vgg', 'squeeze').
        device: Device for computation.

    Returns:
        LPIPS value (0 to 1, lower = more similar).
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips is required. Install with: pip install lpips")

    # LPIPS expects 3-channel images in [-1, 1]
    # Convert grayscale to 3-channel if needed
    if img1.shape[0] == 1:
        img1 = img1.repeat(3, 1, 1)
        img2 = img2.repeat(3, 1, 1)
    
    # Ensure in [-1, 1] range (assume input is [0, 1])
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1
    
    # Add batch dimension
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    
    # Initialize LPIPS model (cached for efficiency)
    if not hasattr(compute_lpips, "_model") or compute_lpips._net != net:
        compute_lpips._model = lpips.LPIPS(net=net).to(device)
        compute_lpips._net = net
    
    with torch.no_grad():
        lpips_val = compute_lpips._model(img1, img2)
    
    return lpips_val.item()


def load_image(path: str, spatial_dims: int = 2) -> torch.Tensor:
    """Load an image as a tensor.

    Args:
        path: Path to image file.
        spatial_dims: 2 for 2D, 3 for 3D.

    Returns:
        Tensor of shape (C, H, W) or (C, D, H, W) in [0, 1] range.
    """
    path_lower = path.lower()
    
    if spatial_dims == 2:
        from PIL import Image
        img = Image.open(path).convert("L")  # Grayscale
        img_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
    else:
        # 3D volume
        if path_lower.endswith(".npy"):
            volume = np.load(path).astype(np.float32)
        elif path_lower.endswith((".nii", ".nii.gz")):
            import nibabel as nib
            nii = nib.load(path)
            volume = nii.get_fdata().astype(np.float32)
        else:
            raise ValueError(f"Unsupported 3D format: {path}")
        
        # Normalize to [0, 1]
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)
        
        return torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)


def pair_images_by_filename(
    real_dir: str,
    fake_dir: str,
    spatial_dims: int = 2,
) -> list[tuple[str, str]]:
    """Find paired images by matching filenames.

    Args:
        real_dir: Directory or JSON file with real images.
        fake_dir: Directory with generated images.
        spatial_dims: 2 for 2D, 3 for 3D.

    Returns:
        List of (real_path, fake_path) tuples.
    """
    fake_dir = Path(fake_dir)
    pairs = []

    # Get real image paths
    if real_dir.endswith(".json"):
        # Load from JSON (DTT format)
        with open(real_dir) as f:
            data = json.load(f)
        real_paths = {Path(entry["image"]).stem: entry["image"] for entry in data}
    else:
        # Load from directory
        extensions = (".png", ".jpg", ".jpeg") if spatial_dims == 2 else (".nii", ".nii.gz", ".npy")
        real_paths = {}
        for ext in extensions:
            for p in Path(real_dir).glob(f"*{ext}"):
                real_paths[p.stem] = str(p)

    # Match with fake images
    fake_extensions = (".png", ".jpg", ".jpeg") if spatial_dims == 2 else (".nii", ".nii.gz", ".npy")
    for ext in fake_extensions:
        for fake_path in fake_dir.glob(f"*{ext}"):
            name = fake_path.stem
            if name in real_paths:
                pairs.append((real_paths[name], str(fake_path)))

    return pairs


def evaluate_paired_images(
    real_dir: str,
    fake_dir: str,
    spatial_dims: int = 2,
    compute_ssim_metric: bool = True,
    compute_lpips_metric: bool = True,
    device: str = "cuda",
    target_size: tuple[int, ...] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate paired images using SSIM and LPIPS.

    Args:
        real_dir: Directory or JSON file with real images.
        fake_dir: Directory with generated images.
        spatial_dims: 2 for 2D, 3 for 3D.
        compute_ssim_metric: Whether to compute SSIM.
        compute_lpips_metric: Whether to compute LPIPS (2D only).
        device: Device for LPIPS computation.
        target_size: Optional resize target (H, W) or (D, H, W).
        output_path: Optional path to save results.

    Returns:
        Dictionary with mean and std for each metric.
    """
    from dtt.utils.logging import get_console
    console = get_console()

    # Find paired images
    pairs = pair_images_by_filename(real_dir, fake_dir, spatial_dims)
    
    if not pairs:
        console.log("[bold red]Error:[/bold red] No paired images found. "
                   "Make sure generated images have same filenames as originals.")
        return {}

    console.log(f"[cyan]Found {len(pairs)} paired images[/cyan]")

    ssim_values = []
    lpips_values = []

    for real_path, fake_path in tqdm(pairs, desc="Computing paired metrics"):
        try:
            real_img = load_image(real_path, spatial_dims)
            fake_img = load_image(fake_path, spatial_dims)
            
            # Resize if needed
            if target_size is not None:
                if spatial_dims == 2:
                    real_img = torch.nn.functional.interpolate(
                        real_img.unsqueeze(0), size=target_size, mode="bilinear"
                    ).squeeze(0)
                    fake_img = torch.nn.functional.interpolate(
                        fake_img.unsqueeze(0), size=target_size, mode="bilinear"
                    ).squeeze(0)

            # Compute SSIM
            if compute_ssim_metric:
                ssim_val = compute_ssim(real_img, fake_img, spatial_dims=spatial_dims)
                ssim_values.append(ssim_val)

            # Compute LPIPS (2D only)
            if compute_lpips_metric and spatial_dims == 2:
                lpips_val = compute_lpips(real_img, fake_img, device=device)
                lpips_values.append(lpips_val)

        except Exception as e:
            logger.warning(f"Error processing {real_path}: {e}")
            continue

    # Compute statistics
    results = {"n_pairs": len(pairs)}
    
    if ssim_values:
        results["ssim_mean"] = float(np.mean(ssim_values))
        results["ssim_std"] = float(np.std(ssim_values))
        console.print(f"  [bold]SSIM:[/bold] {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")

    if lpips_values:
        results["lpips_mean"] = float(np.mean(lpips_values))
        results["lpips_std"] = float(np.std(lpips_values))
        console.print(f"  [bold]LPIPS:[/bold] {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        console.log(f"[green]Saved paired metrics to:[/green] {output_path}")

    return results
