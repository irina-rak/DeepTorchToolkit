from __future__ import annotations

import os
from typing import Any


def build_callbacks(cfg: dict[str, Any], run_dir: str | None = None) -> list[object]:
    """Construct common Lightning callbacks from config.

    Returns a list of callback instances. Imports are local to avoid heavy deps at import time.
    Config is validated via Pydantic schemas, so no need to check for typos.

    Note: RichProgressBar and RichModelSummary are added by default to use Rich
    instead of tqdm for all progress bars and model summaries.
    
    Args:
        cfg: Configuration dictionary
        run_dir: Directory for this run's outputs (checkpoints, logs, etc.)
    """
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        RichModelSummary,
        RichProgressBar,
    )

    cb_cfg = cfg.get("callbacks", {})

    callbacks: list[object] = []

    # Add Rich progress bar by default (replaces tqdm)
    callbacks.append(RichProgressBar())

    # Add Rich model summary (replaces default ModelSummary)
    callbacks.append(RichModelSummary(max_depth=2))

    mc = cb_cfg.get("model_checkpoint")
    if mc:
        # Set checkpoint directory to run_dir/checkpoints if not specified
        mc_dict = mc.copy() if isinstance(mc, dict) else {}
        if run_dir and "dirpath" not in mc_dict:
            mc_dict["dirpath"] = os.path.join(run_dir, "checkpoints")
        callbacks.append(ModelCheckpoint(**mc_dict))

    es = cb_cfg.get("early_stopping")
    if es:
        callbacks.append(EarlyStopping(**es))

    lr = cb_cfg.get("lr_monitor")
    if lr:
        callbacks.append(LearningRateMonitor(**lr))

    return callbacks
