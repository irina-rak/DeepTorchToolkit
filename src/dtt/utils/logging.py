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


def make_wandb_logger(cfg: dict[str, Any]):
    """Create a WandbLogger from a logger config dict.

    Expects keys under cfg["logger"]["wandb"]: project, name (optional), entity (optional), tags (list), mode.
    Gracefully handles missing API key by allowing offline mode.
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

    return WandbLogger(project=project, name=name, entity=entity, tags=tags, mode=mode)
