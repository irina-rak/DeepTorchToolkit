from __future__ import annotations

import os
from typing import Any

import dtt.data  # noqa: F401 - Import to register datamodules
import dtt.models  # noqa: F401 - Import to register models
from dtt.training.callbacks import build_callbacks
from dtt.utils.logging import get_console, make_wandb_logger
from dtt.utils.registry import get_datamodule, get_model

console = get_console()


def _save_config(cfg: dict[str, Any], run_dir: str) -> str:
    """Save the configuration to the run directory for reproducibility.

    This saves the fully resolved configuration (with all defaults and
    interpolated values) so experiments can be exactly reproduced later.

    Args:
        cfg: Configuration dictionary (fully resolved)
        run_dir: Directory for this run's outputs

    Returns:
        Path to the saved config file
    """
    import yaml

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    console.log(f"[bold green]✓[/bold green] Saved config to: {config_path}")
    return config_path


def _setup_output_directory(cfg: dict[str, Any]) -> str:
    """Create and return the structured output directory for this run.

    Directory structure: <save_dir>/<project>/<run_name_with_timestamp>/

    Args:
        cfg: Configuration dictionary

    Returns:
        Absolute path to the run directory
    """
    from datetime import datetime

    base_dir = cfg.get("save_dir", "experiments")
    logger_cfg = cfg.get("logger", {}).get("wandb", {})
    project_name = logger_cfg.get("project", "default")
    run_name = logger_cfg.get("name", "run")

    # Add timestamp to run name if not already present
    if run_name and "_202" not in run_name:  # Simple check if timestamp already added
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_name}_{timestamp}"

    # Structure: save_dir/project/run_name
    run_dir = os.path.abspath(os.path.join(base_dir, project_name, run_name))
    os.makedirs(run_dir, exist_ok=True)

    console.log(f"[bold green]Output directory:[/bold green] {run_dir}")

    return run_dir


def _build_trainer(cfg: dict[str, Any], run_dir: str):
    from lightning.pytorch import Trainer

    # Override trainer's default_root_dir with our structured directory
    trainer_cfg = cfg.get("trainer", {}).copy()
    trainer_cfg["default_root_dir"] = run_dir

    trainer = Trainer(callbacks=build_callbacks(cfg, run_dir), **trainer_cfg)
    return trainer


def run_training(cfg: dict[str, Any]) -> None:
    """Assemble and run the Trainer based on the config dict (already validated)."""
    # Setup structured output directory
    run_dir = _setup_output_directory(cfg)

    # Save config for reproducibility
    _save_config(cfg, run_dir)

    # Build logger with the run directory
    logger = make_wandb_logger(cfg, save_dir=run_dir)

    # Instantiate data + model from registries
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    datamodule = get_datamodule(data_cfg.get("name", "medical2d"))(cfg)
    model = get_model(model_cfg.get("name", "monai.unet"))(cfg)

    trainer = _build_trainer(cfg, run_dir)
    trainer.logger = logger

    trainer.fit(model=model, datamodule=datamodule)
