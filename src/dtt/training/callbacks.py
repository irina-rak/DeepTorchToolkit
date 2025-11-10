from __future__ import annotations

from typing import Any, Dict, List


def build_callbacks(cfg: Dict[str, Any]) -> List[object]:
    """Construct common Lightning callbacks from config.

    Returns a list of callback instances. Imports are local to avoid heavy deps at import time.
    Config is validated via Pydantic schemas, so no need to check for typos.
    """
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    cb_cfg = cfg.get("callbacks", {})

    callbacks: List[object] = []

    mc = cb_cfg.get("model_checkpoint")
    if mc:
        callbacks.append(ModelCheckpoint(**mc))

    es = cb_cfg.get("early_stopping")
    if es:
        callbacks.append(EarlyStopping(**es))

    lr = cb_cfg.get("lr_monitor")
    if lr:
        callbacks.append(LearningRateMonitor(**lr))

    return callbacks
