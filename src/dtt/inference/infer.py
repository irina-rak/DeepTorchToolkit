"""Inference/generation script that leverages test_step from Lightning modules.

This module provides a unified inference interface that works with any DTT model
by using the model's test_step method through the Lightning Trainer.test() API.
"""

from __future__ import annotations

import os
from typing import Any

import dtt.data  # noqa: F401 - Import to register datamodules
import dtt.models  # noqa: F401 - Import to register models
from dtt.utils.logging import get_console
from dtt.utils.registry import get_datamodule, get_model

console = get_console()


def _setup_output_directory(cfg: dict[str, Any]) -> str:
    """Create and return the output directory for inference.

    Directory structure: <output_dir>/generated_samples/
    Evaluation results would go to: <output_dir>/evaluation/

    Args:
        cfg: Configuration dictionary

    Returns:
        Absolute path to the output directory
    """
    output_dir = os.path.abspath(cfg.get("output_dir", "outputs"))
    os.makedirs(output_dir, exist_ok=True)

    console.log(f"[bold green]Output directory:[/bold green] {output_dir}")

    return output_dir


def _load_checkpoint(checkpoint_path: str, cfg: dict[str, Any]):
    """Load model from checkpoint.

    This function attempts to load model configuration from the checkpoint's
    saved hyperparameters first, falling back to the provided config if not available.
    This ensures the model architecture matches what was trained.

    Args:
        checkpoint_path: Path to checkpoint file
        cfg: Configuration dictionary (used as fallback and for non-model settings)

    Returns:
        Loaded Lightning module, updated config dict
    """
    import torch

    console.log(f"[bold cyan]Loading checkpoint:[/bold cyan] {checkpoint_path}")

    # Try to load the checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint state
    # Note: weights_only=False is needed for Lightning checkpoints which contain
    # more than just weights (hyperparameters, optimizer state, etc.)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Try to extract model config from checkpoint's hyperparameters
    hparams = checkpoint.get("hyper_parameters", {})
    saved_config = hparams.get("config") if hparams else None

    if saved_config is not None:
        console.log(
            "[green]✓ Found saved config in checkpoint, using for model architecture[/green]"
        )

        # Convert saved Config object to dict if needed
        if hasattr(saved_config, "model_dump"):
            saved_cfg_dict = saved_config.model_dump()
        elif hasattr(saved_config, "dict"):
            saved_cfg_dict = saved_config.dict()
        else:
            saved_cfg_dict = dict(saved_config) if hasattr(saved_config, "__iter__") else {}

        # Use saved model config for architecture, but keep user's inference settings
        if saved_cfg_dict:
            # Preserve user's settings for runtime (not model architecture)
            user_inference = cfg.get("inference", {})
            user_output_dir = cfg.get("output_dir")
            user_batch_size = cfg.get("data", {}).get("batch_size")
            user_trainer = cfg.get("trainer", {})
            user_project = cfg.get("project")
            user_run_name = cfg.get("run_name")
            user_solver_args = cfg.get("model", {}).get("params", {}).get("solver_args")

            # Use saved config as base (for model architecture)
            cfg = saved_cfg_dict

            # Overlay user's runtime settings
            if user_inference:
                cfg["inference"] = {**cfg.get("inference", {}), **user_inference}
            if user_output_dir:
                cfg["output_dir"] = user_output_dir
            if user_batch_size and "data" in cfg:
                cfg["data"]["batch_size"] = user_batch_size
            if user_trainer:
                cfg["trainer"] = user_trainer  # Use user's trainer config entirely
            if user_project:
                cfg["project"] = user_project
            if user_run_name:
                cfg["run_name"] = user_run_name
            if user_solver_args:
                # Override solver settings for inference
                if "model" in cfg and "params" in cfg["model"]:
                    cfg["model"]["params"]["solver_args"] = user_solver_args

            console.log(f"  Model: {cfg.get('model', {}).get('name', 'unknown')}")
    else:
        console.log("[yellow]⚠ No saved config in checkpoint, using provided config[/yellow]")

    # Get model builder from registry
    model_cfg = cfg.get("model", {})
    model_builder = get_model(model_cfg.get("name", "monai.unet"))

    # Build model from config
    model = model_builder(cfg)

    # Load state dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    console.log("[bold green]✓[/bold green] Checkpoint loaded successfully")

    return model, cfg


def _build_trainer(cfg: dict[str, Any], output_dir: str):
    """Build Lightning Trainer for inference.

    Args:
        cfg: Configuration dictionary
        output_dir: Output directory for results

    Returns:
        Lightning Trainer instance
    """
    from lightning.pytorch import Trainer

    # Build trainer config for inference
    trainer_cfg = cfg.get("trainer", {}).copy()

    # Override settings for inference
    trainer_cfg["default_root_dir"] = output_dir
    # Disable training-specific features
    trainer_cfg.setdefault("logger", False)  # Disable logging by default for inference
    trainer_cfg.setdefault("enable_checkpointing", False)

    # Remove None values that cause Lightning errors
    trainer_cfg = {k: v for k, v in trainer_cfg.items() if v is not None}

    console.log(f"[bold cyan]Trainer config:[/bold cyan] {trainer_cfg}")

    trainer = Trainer(**trainer_cfg)
    return trainer


def run_inference(cfg: dict[str, Any], checkpoint_path: str) -> None:
    """Run inference using the model's test_step method.

    This function:
    1. Loads a model from checkpoint (and extracts saved config if available)
    2. Sets up the datamodule (or creates a dummy one for unconditional generation)
    3. Runs Trainer.test() which calls test_step on each batch
    4. test_step implementations handle generation, metrics, and saving

    Args:
        cfg: Configuration dictionary (used for inference settings, model config from checkpoint preferred)
        checkpoint_path: Path to model checkpoint
    """
    # Load model from checkpoint (this may update cfg with saved config)
    model, cfg = _load_checkpoint(checkpoint_path, cfg)

    # Setup output directory (after loading checkpoint so we have the right config)
    output_dir = _setup_output_directory(cfg)

    # Set model to inference mode
    model.eval()

    # Check if we're running without test data (unconditional generation)
    inference_cfg = cfg.get("inference", {})
    use_test_data = inference_cfg.get("use_test_data", True)
    save_comparison = inference_cfg.get("save_comparison", True)

    if use_test_data:
        # Standard mode: use test datamodule
        data_cfg = cfg.get("data", {})
        datamodule = get_datamodule(data_cfg.get("name", "medical2d"))(cfg)
        console.log("[bold cyan]Mode:[/bold cyan] Inference with test data")
    else:
        # Unconditional mode: create a dummy datamodule
        from dtt.inference.dummy_datamodule import DummyDataModule

        num_batches = inference_cfg.get("num_batches", 10)
        batch_size = cfg.get("data", {}).get("batch_size", 16)

        # Get spatial dimensions from model config
        model_cfg = cfg.get("model", {})
        params = model_cfg.get("params", {})
        unet_config = params.get("unet_config", {})
        spatial_dims = unet_config.get("spatial_dims", params.get("spatial_dims", 2))
        in_channels = unet_config.get("in_channels", 1)
        spatial_size = cfg.get("data", {}).get("params", {}).get("spatial_size", [128, 128])

        datamodule = DummyDataModule(
            num_batches=num_batches,
            batch_size=batch_size,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            spatial_size=spatial_size,
        )
        console.log(
            f"[bold cyan]Mode:[/bold cyan] Unconditional generation ({num_batches} batches)"
        )
        # Unconditional mode never saves comparisons
        save_comparison = False

    # Build trainer
    trainer = _build_trainer(cfg, output_dir)

    # Store inference config in model for test_step to access
    model.inference_output_dir = output_dir
    model.inference_mode = "unconditional" if not use_test_data else "conditional"
    model.inference_save_comparison = save_comparison

    console.log("[bold green]Starting inference...[/bold green]")

    # Run inference using test() - this calls test_step on each batch
    trainer.test(model=model, datamodule=datamodule)

    console.log(f"[bold green]✓ Inference complete![/bold green] Results saved to: {output_dir}")
