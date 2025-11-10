from __future__ import annotations

from typing import Any

from dtt.training.callbacks import build_callbacks
from dtt.utils.logging import make_wandb_logger
from dtt.utils.registry import get_datamodule, get_model


def _build_trainer(cfg: dict[str, Any]):
    from lightning.pytorch import Trainer

    trainer = Trainer(callbacks=build_callbacks(cfg), **cfg.get("trainer", {}))
    return trainer


def run_training(cfg: dict[str, Any]) -> None:
    """Assemble and run the Trainer based on the config dict (already validated)."""
    # Build logger
    logger = make_wandb_logger(cfg)
    # Instantiate data + model from registries
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    datamodule = get_datamodule(data_cfg.get("name", "medical2d"))(cfg)
    model = get_model(model_cfg.get("name", "monai.unet"))(cfg)

    trainer = _build_trainer(cfg)
    trainer.logger = logger

    trainer.fit(model=model, datamodule=datamodule)
