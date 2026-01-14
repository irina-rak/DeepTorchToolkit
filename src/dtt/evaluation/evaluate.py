"""Main evaluation script for generative models.

This module provides the main evaluation functionality for comparing
generated images against real images using distribution-based metrics.

It handles:
- Loading images from directories (PNG, NIfTI, NPY formats)
- Loading images from JSON files (DTT dataset format)
- Feature extraction using appropriate 2D or 3D extractors
- Computing FID and KID metrics
- Generating evaluation reports

Example usage:
    python -m dtt.evaluation.evaluate \\
        --real-dir /path/to/real \\
        --fake-dir /path/to/generated \\
        --spatial-dims 3 \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from dtt.evaluation.feature_extractors import get_feature_extractor
from dtt.evaluation.metrics import compute_distribution_metrics

logger = logging.getLogger(__name__)


def load_image_2d(path: str) -> np.ndarray:
    """Load a 2D image from file.

    Args:
        path: Path to image file (PNG, JPG, etc.)

    Returns:
        Image array of shape (H, W) or (H, W, C) with values in [0, 1].
    """
    from PIL import Image

    img = Image.open(path)
    img_array = np.array(img).astype(np.float32)

    # Normalize to [0, 1]
    if img_array.max() > 1.0:
        img_array = img_array / 255.0

    return img_array


def load_image_3d(path: str) -> np.ndarray:
    """Load a 3D volume from file.

    Args:
        path: Path to volume file (NIfTI, NPY, etc.)

    Returns:
        Volume array of shape (D, H, W) or (C, D, H, W) with values in [0, 1].
    """
    path_lower = path.lower()

    if path_lower.endswith(".npy"):
        volume = np.load(path).astype(np.float32)
    elif path_lower.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError(
                "nibabel is required for loading NIfTI files. "
                "Install it with: pip install nibabel"
            ) from e
        nii = nib.load(path)
        volume = nii.get_fdata().astype(np.float32)
    else:
        raise ValueError(f"Unsupported 3D format: {path}")

    # Normalize to [0, 1]
    v_min, v_max = volume.min(), volume.max()
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume


def load_images_from_directory(
    directory: str,
    spatial_dims: int = 2,
    max_samples: int | None = None,
    extensions_2d: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    extensions_3d: tuple[str, ...] = (".nii", ".nii.gz", ".npy"),
    target_size: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Load images from a directory into a batch tensor.

    Args:
        directory: Path to directory containing images.
        spatial_dims: Number of spatial dimensions (2 or 3).
        max_samples: Maximum number of samples to load. None for all.
        extensions_2d: File extensions to look for in 2D mode.
        extensions_3d: File extensions to look for in 3D mode.
        target_size: Optional target size for resizing (H, W) or (D, H, W).

    Returns:
        Tensor of shape (N, C, H, W) for 2D or (N, C, D, H, W) for 3D.
        Values are in [0, 1].

    Raises:
        ValueError: If no images found in directory.
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    extensions = extensions_2d if spatial_dims == 2 else extensions_3d

    # Find all image files
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(str(directory / f"*{ext}")))
        image_paths.extend(glob(str(directory / f"**/*{ext}"), recursive=True))

    # Remove duplicates and sort
    image_paths = sorted(set(image_paths))

    if not image_paths:
        raise ValueError(
            f"No images found in {directory} with extensions {extensions}"
        )

    if max_samples is not None:
        image_paths = image_paths[:max_samples]

    logger.info(f"Loading {len(image_paths)} images from {directory}")

    # Load images
    images = []
    load_fn = load_image_2d if spatial_dims == 2 else load_image_3d

    for path in tqdm(image_paths, desc="Loading images"):
        try:
            img = load_fn(path)
            images.append(img)
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")

    if not images:
        raise ValueError(f"Failed to load any images from {directory}")

    # Convert to tensor
    images_tensor = _images_to_tensor(images, spatial_dims, target_size=target_size)

    return images_tensor


def load_images_from_json(
    json_path: str,
    spatial_dims: int = 2,
    max_samples: int | None = None,
    image_key: str = "image",
    target_size: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Load images from a JSON dataset file (DTT format).

    This function loads images using the same JSON format as DTT dataloaders,
    making evaluation consistent with training/testing data splits.

    Args:
        json_path: Path to JSON file containing list of dicts with image paths.
        spatial_dims: Number of spatial dimensions (2 or 3).
        max_samples: Maximum number of samples to load. None for all.
        image_key: Key in JSON entries containing the image path (default: "image").
        target_size: Optional target size for resizing (H, W) or (D, H, W).

    Returns:
        Tensor of shape (N, C, H, W) for 2D or (N, C, D, H, W) for 3D.
        Values are in [0, 1].

    Example JSON format:
        [
            {"name": "sample1", "image": "path/to/image1.png"},
            {"name": "sample2", "image": "path/to/image2.png"},
            ...
        ]
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise ValueError(f"JSON file does not exist: {json_path}")

    # Load JSON
    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of entries, got {type(data)}")

    if max_samples is not None:
        data = data[:max_samples]

    logger.info(f"Loading {len(data)} images from {json_path}")

    # Extract image paths
    image_paths = []
    for entry in data:
        if image_key not in entry:
            raise ValueError(f"Entry missing '{image_key}' key: {entry}")
        image_paths.append(entry[image_key])

    # Load images
    images = []
    load_fn = load_image_2d if spatial_dims == 2 else load_image_3d

    for path in tqdm(image_paths, desc="Loading images from JSON"):
        try:
            img = load_fn(path)
            images.append(img)
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")

    if not images:
        raise ValueError(f"Failed to load any images from {json_path}")

    # Convert to tensor
    images_tensor = _images_to_tensor(images, spatial_dims, target_size=target_size)

    return images_tensor


def load_images(
    source: str,
    spatial_dims: int = 2,
    max_samples: int | None = None,
    target_size: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Load images from either a directory or JSON file.

    Automatically detects whether source is a JSON file or directory.

    Args:
        source: Path to directory or JSON file.
        spatial_dims: Number of spatial dimensions (2 or 3).
        max_samples: Maximum number of samples to load.
        target_size: Optional target size to resize all images to (H, W) or (D, H, W).

    Returns:
        Tensor of shape (N, C, H, W) for 2D or (N, C, D, H, W) for 3D.
    """
    source_path = Path(source)
    
    if source.endswith(".json") or source_path.suffix == ".json":
        return load_images_from_json(source, spatial_dims, max_samples, target_size=target_size)
    else:
        return load_images_from_directory(source, spatial_dims, max_samples, target_size=target_size)


def _images_to_tensor(
    images: list[np.ndarray],
    spatial_dims: int,
    target_size: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Convert a list of numpy arrays to a batched tensor.

    Handles varying image sizes by resizing to target_size if provided,
    otherwise resizes to the first image's size.

    Args:
        images: List of numpy arrays.
        spatial_dims: 2 or 3.
        target_size: Optional target size for resizing (H, W) or (D, H, W).
                     If None, uses first image size when shapes vary.

    Returns:
        Batched tensor of shape (N, C, ...).
    """
    # Convert each image to proper shape
    processed = []

    for img in images:
        # Ensure channel dimension
        if spatial_dims == 2:
            if img.ndim == 2:
                img = img[np.newaxis, ...]  # (H, W) -> (1, H, W)
            elif img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
                img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            # If RGB/RGBA, convert to grayscale for medical images
            if img.shape[0] == 3:
                img = img.mean(axis=0, keepdims=True)
            elif img.shape[0] == 4:
                img = img[:3].mean(axis=0, keepdims=True)
        else:  # 3D
            if img.ndim == 3:
                img = img[np.newaxis, ...]  # (D, H, W) -> (1, D, H, W)
            elif img.ndim == 4 and img.shape[-1] in [1, 3]:
                img = np.transpose(img, (3, 0, 1, 2))  # (D, H, W, C) -> (C, D, H, W)

        processed.append(torch.from_numpy(img).float())

    # Determine target shape for resizing
    if target_size is not None:
        # User-specified target size
        resize_target = tuple(target_size)
        logger.info(f"Resizing all images to target size: {resize_target}")
    else:
        # Check if all images have the same shape
        shapes = [img.shape for img in processed]
        if len(set(shapes)) > 1:
            logger.warning(
                f"Images have varying shapes. Resizing to first image shape: {shapes[0]}"
            )
            resize_target = processed[0].shape[1:]  # Exclude channel dim
        else:
            resize_target = None  # No resizing needed

    # Resize if needed
    if resize_target is not None:
        resized = []
        for img in processed:
            if img.shape[1:] != resize_target:
                if spatial_dims == 2:
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=resize_target,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                else:
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=resize_target,
                        mode="trilinear",
                        align_corners=False,
                    ).squeeze(0)
            resized.append(img)
        processed = resized

    return torch.stack(processed)


def extract_features(
    images: torch.Tensor,
    feature_extractor: torch.nn.Module,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Extract features from images using the given extractor.

    Args:
        images: Tensor of shape (N, C, ...).
        feature_extractor: Feature extractor module.
        batch_size: Batch size for feature extraction.
        device: Device to run extraction on.

    Returns:
        Feature tensor of shape (N, feature_dim).
    """
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        logger.info("CUDA not available, using CPU for feature extraction")

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    features_list = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx].to(device)

            batch_features = feature_extractor(batch)
            features_list.append(batch_features.cpu())

    return torch.cat(features_list, dim=0)


def evaluate_generated_images(
    real_dir: str,
    fake_dir: str,
    spatial_dims: int = 2,
    feature_extractor: Literal["auto", "inception", "medicalnet"] = "auto",
    max_samples: int | None = None,
    batch_size: int = 32,
    device: str = "cuda",
    compute_kid: bool = True,
    output_path: str | None = None,
    target_size: tuple[int, ...] | list[int] | None = None,
    save_visualizations: bool = False,
    visualization_dir: str | None = None,
    plot_tsne: bool = True,
    plot_sample_grid: bool = True,
    plot_histogram: bool = True,
    num_grid_samples: int = 16,
) -> dict[str, float]:
    """Evaluate generated images against real images.

    Main evaluation function that:
    1. Loads real and generated images from directories or JSON files
    2. Extracts features using appropriate 2D or 3D extractor
    3. Computes FID and optionally KID metrics
    4. Optionally generates visualization plots
    5. Optionally saves results to JSON

    Args:
        real_dir: Path to real images (directory OR JSON file).
        fake_dir: Path to generated images (directory OR JSON file).
        spatial_dims: Number of spatial dimensions (2 or 3).
        feature_extractor: Type of feature extractor to use.
        max_samples: Maximum samples to load (None for all).
        batch_size: Batch size for feature extraction.
        device: Device for computation.
        compute_kid: Whether to compute KID in addition to FID.
        output_path: Optional path to save results as JSON.
        target_size: Optional target size to resize all images (H, W) or (D, H, W).
        save_visualizations: Whether to generate visualization plots.
        visualization_dir: Directory for visualization plots (defaults to output_path parent).
        plot_tsne: Whether to generate t-SNE plot.
        plot_sample_grid: Whether to generate sample grid.
        plot_histogram: Whether to generate intensity histogram.
        num_grid_samples: Number of samples per side in sample grid.

    Returns:
        Dictionary containing evaluation metrics:
            - fid: Fréchet Inception Distance
            - kid_mean: KID mean (if compute_kid=True)
            - kid_std: KID std (if compute_kid=True)
            - n_real: Number of real samples
            - n_fake: Number of fake samples
    """
    from dtt.utils.logging import get_console

    console = get_console()

    # Detect source type for logging
    real_type = "JSON" if real_dir.endswith(".json") else "directory"
    fake_type = "JSON" if fake_dir.endswith(".json") else "directory"

    if target_size:
        console.log(f"[cyan]Target image size:[/cyan] {target_size}")

    console.log(f"[cyan]Loading real images from {real_type}:[/cyan] {real_dir}")
    real_images = load_images(real_dir, spatial_dims=spatial_dims, max_samples=max_samples, target_size=target_size)
    console.log(f"[green]Loaded {len(real_images)} real images[/green]")

    console.log(f"[cyan]Loading fake images from {fake_type}:[/cyan] {fake_dir}")
    fake_images = load_images(fake_dir, spatial_dims=spatial_dims, max_samples=max_samples, target_size=target_size)
    console.log(f"[green]Loaded {len(fake_images)} fake images[/green]")

    # Get feature extractor
    console.log(f"[cyan]Initializing feature extractor:[/cyan] {feature_extractor}")
    extractor = get_feature_extractor(
        spatial_dims=spatial_dims,
        extractor_type=feature_extractor,
    )
    console.log(f"[green]Feature dimension:[/green] {extractor.output_dim}")

    # Extract features
    console.log("[cyan]Extracting features from real images...[/cyan]")
    real_features = extract_features(
        real_images, extractor, batch_size=batch_size, device=device
    )

    console.log("[cyan]Extracting features from fake images...[/cyan]")
    fake_features = extract_features(
        fake_images, extractor, batch_size=batch_size, device=device
    )

    # Compute metrics
    console.log("[cyan]Computing distribution metrics...[/cyan]")
    results = compute_distribution_metrics(
        real_features, fake_features, compute_kid_metric=compute_kid
    )

    # Print results
    console.print("\n[bold green]═══ Evaluation Results ═══[/bold green]")
    console.print(f"  [bold]FID:[/bold] {results['fid']:.4f}")
    if compute_kid:
        console.print(
            f"  [bold]KID:[/bold] {results['kid_mean']:.6f} ± {results['kid_std']:.6f}"
        )
    console.print(f"  [dim]Real samples: {results['n_real']}[/dim]")
    console.print(f"  [dim]Fake samples: {results['n_fake']}[/dim]")
    console.print("[bold green]═══════════════════════════[/bold green]\n")

    # Generate visualizations if requested
    if save_visualizations:
        from dtt.evaluation.visualize import generate_evaluation_visualizations
        
        # Determine visualization directory
        if visualization_dir:
            viz_dir = visualization_dir
        elif output_path:
            viz_dir = str(Path(output_path).parent / "visualizations")
        else:
            viz_dir = "outputs/evaluation/visualizations"
        
        console.log(f"[cyan]Generating visualizations in:[/cyan] {viz_dir}")
        saved_plots = generate_evaluation_visualizations(
            real_images=real_images,
            fake_images=fake_images,
            real_features=real_features,
            fake_features=fake_features,
            output_dir=viz_dir,
            plot_tsne=plot_tsne,
            plot_sample_grid=plot_sample_grid,
            plot_histogram=plot_histogram,
            num_grid_samples=num_grid_samples,
        )
        console.log(f"[green]Saved {len(saved_plots)} visualization plots[/green]")
        results["visualizations"] = saved_plots

    # Save results if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        results["metadata"] = {
            "real_dir": str(real_dir),
            "fake_dir": str(fake_dir),
            "spatial_dims": spatial_dims,
            "feature_extractor": feature_extractor,
            "target_size": list(target_size) if target_size else None,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        console.log(f"[green]Results saved to:[/green] {output_path}")

    return results


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated images using FID and KID metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate 2D images
  python -m dtt.evaluation.evaluate --real-dir /path/to/real --fake-dir /path/to/generated

  # Evaluate 3D volumes
  python -m dtt.evaluation.evaluate --real-dir /path/to/real --fake-dir /path/to/generated --spatial-dims 3

  # Save results to JSON
  python -m dtt.evaluation.evaluate --real-dir /path/to/real --fake-dir /path/to/generated --output results.json
""",
    )

    parser.add_argument(
        "--real-dir",
        type=str,
        required=True,
        help="Directory or JSON file containing real images",
    )
    parser.add_argument(
        "--fake-dir",
        type=str,
        required=True,
        help="Directory or JSON file containing generated/fake images",
    )
    parser.add_argument(
        "--spatial-dims",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of spatial dimensions (2 for images, 3 for volumes)",
    )
    parser.add_argument(
        "--feature-extractor",
        type=str,
        default="auto",
        choices=["auto", "inception", "medicalnet"],
        help="Feature extractor type (auto selects based on spatial-dims)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cuda or cpu)",
    )
    parser.add_argument(
        "--no-kid",
        action="store_true",
        help="Skip KID computation (faster, FID only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs="+",
        default=None,
        help="Target size to resize images (e.g., 128 128 for 2D)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run evaluation
    evaluate_generated_images(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        spatial_dims=args.spatial_dims,
        feature_extractor=args.feature_extractor,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        compute_kid=not args.no_kid,
        output_path=args.output,
        target_size=tuple(args.target_size) if args.target_size else None,
    )


if __name__ == "__main__":
    main()
