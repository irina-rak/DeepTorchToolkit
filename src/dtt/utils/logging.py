from __future__ import annotations

import os
from typing import Any

from rich.console import Console

_console: Console | None = None


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(highlight=False)
    return _console


def make_wandb_logger(cfg: dict[str, Any], save_dir: str | None = None):
    """Create a WandbLogger from a logger config dict.

    Expects keys under cfg["logger"]["wandb"]:
        - project: W&B project name
        - name: Run name (optional)
        - entity: W&B entity/team (optional)
        - tags: List of tags (optional)
        - mode: "online" or "offline" (default: "offline")
        - api_key: W&B API key (optional, overrides WANDB_API_KEY env var)

    The API key can be provided in three ways (priority order):
        1. In config: logger.wandb.api_key
        2. Environment variable: WANDB_API_KEY
        3. Pre-authenticated via `wandb login`
        
    Args:
        cfg: Configuration dictionary
        save_dir: Directory where WandB should save its files (optional)
    """
    from lightning.pytorch.loggers import (
        WandbLogger,  # local import to avoid hard dep at import time
    )

    wandb_cfg = cfg.get("logger", {}).get("wandb", {})
    mode = wandb_cfg.get("mode", os.getenv("WANDB_MODE", "offline"))
    project = wandb_cfg.get("project", "dtt")
    name = wandb_cfg.get("name")
    entity = wandb_cfg.get("entity")
    tags = wandb_cfg.get("tags", [])
    api_key = wandb_cfg.get("api_key")

    # Set API key if provided in config
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        console = get_console()
        console.log("[bold green]✓[/bold green] W&B API key set from config")

    # Create logger with save_dir if provided
    logger_kwargs = {
        "project": project,
        "name": name,
        "entity": entity,
        "tags": tags,
        "mode": mode,
    }
    
    if save_dir:
        logger_kwargs["save_dir"] = save_dir
    
    return WandbLogger(**logger_kwargs)
