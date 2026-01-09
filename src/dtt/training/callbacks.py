from __future__ import annotations

import os
from typing import Any

from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    """Exponential Moving Average (EMA) callback using swa_utils.AveragedModel.

    This callback properly handles both parameters AND buffers (like BatchNorm
    running statistics), unlike naive EMA implementations that only average
    parameters.

    The EMA model is exposed to the LightningModule via the `ema_model` attribute,
    which can be accessed during validation/inference for sample generation.

    Args:
        decay: EMA decay rate (0.9999 is typical for diffusion models)
        update_every: Update EMA every N training steps (default: 10).
            Updating every 10-50 steps saves compute with negligible quality impact.
        update_bn_on_train_end: Whether to run forward pass on training data
            to update BatchNorm running statistics at end of training.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        update_every: int = 10,
        update_bn_on_train_end: bool = True,
    ):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.update_bn_on_train_end = update_bn_on_train_end
        self.ema_model = None

    def on_fit_start(self, trainer, pl_module):
        """Create EMA model at the start of training."""
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        # Get the underlying model to wrap
        # Support both pl_module.model (our convention) and direct pl_module
        model = getattr(pl_module, "model", pl_module)

        # Create EMA model with efficient multi_avg_fn
        # Critical: use_buffers=True ensures BatchNorm running stats are updated/copied
        # Without this, BN stats stay at random init, causing garbage output
        self.ema_model = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(self.decay),
            device=pl_module.device,
            use_buffers=True,
        )

        # Expose EMA model to LightningModule for inference
        pl_module.ema_model = self.ema_model

        # Log configuration
        from dtt.utils.logging import get_console

        console = get_console()
        console.log(
            f"[bold cyan]EMA enabled:[/bold cyan] decay={self.decay}, "
            f"update_every={self.update_every}"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA model parameters every N steps."""
        if trainer.global_step % self.update_every == 0:
            # Get the underlying model
            model = getattr(pl_module, "model", pl_module)
            self.ema_model.update_parameters(model)

    def on_train_end(self, trainer, pl_module):
        """Optionally update BatchNorm statistics at end of training."""
        if not self.update_bn_on_train_end:
            return

        # Check if model has any BatchNorm layers
        import torch.nn as nn

        has_bn = any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for m in self.ema_model.modules()
        )

        if not has_bn:
            return

        from dtt.utils.logging import get_console

        console = get_console()
        console.log("[cyan]Updating BatchNorm statistics for EMA model...[/cyan]")

        try:
            from torch.optim.swa_utils import update_bn

            train_loader = trainer.datamodule.train_dataloader()
            update_bn(train_loader, self.ema_model, device=pl_module.device)
            console.log("[bold green]✓[/bold green] BatchNorm statistics updated")
        except Exception as e:
            console.log(f"[yellow]Warning: Could not update BN stats: {e}[/yellow]")


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

        # Sanitize filename to prevent directory creation from metric names with "/"
        # Replace {metric/name} with {metric_name} in the filename pattern
        if "filename" in mc_dict and isinstance(mc_dict["filename"], str):
            # Replace metric names with slashes to use underscores instead
            mc_dict["filename"] = mc_dict["filename"].replace("val/loss", "val_loss")

        callbacks.append(ModelCheckpoint(**mc_dict))

    es = cb_cfg.get("early_stopping")
    if es is not None:  # Only add if explicitly configured (not None)
        callbacks.append(EarlyStopping(**es))

    lr = cb_cfg.get("lr_monitor")
    if lr:
        callbacks.append(LearningRateMonitor(**lr))

    # EMA callback for Exponential Moving Average of model weights
    ema = cb_cfg.get("ema")
    if ema:
        ema_dict = ema.copy() if isinstance(ema, dict) else {}
        callbacks.append(EMACallback(**ema_dict))

    return callbacks
