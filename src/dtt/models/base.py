from __future__ import annotations

from typing import Any, Dict, Optional

from dtt.config.schemas import OptimConfig, SchedulerConfig


class BaseLightningModule:
    """A minimal Lightning-like interface to enable type hints without hard dependency at import time.

    Subclasses should inherit from `lightning.pytorch.LightningModule` in practice. This base exists
    to avoid importing Lightning when modules are inspected without heavy deps.
    """

    def __init__(self) -> None:
        pass

    # Lightning will override; present to satisfy type checkers
    def configure_optimizers(self):  # pragma: no cover - only used when Lightning is installed
        raise NotImplementedError


def build_optimizer(params, optim_cfg: OptimConfig):
    """Build optimizer from config."""
    name = optim_cfg.name.lower()
    lr = optim_cfg.lr
    wd = optim_cfg.weight_decay
    if name in {"adam", "adamw"}:
        import torch.optim as optim

        cls = optim.AdamW if name == "adamw" else optim.Adam
        return cls(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        import torch.optim as optim

        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {optim_cfg.name}")


def build_scheduler(optimizer, scheduler_cfg: SchedulerConfig) -> Optional[object]:
    """Build LR scheduler from config.

    Returns None if scheduler is not configured.
    """
    if not scheduler_cfg.name:
        return None

    import torch.optim.lr_scheduler as sched

    name = scheduler_cfg.name.lower()
    params = scheduler_cfg.params

    if name == "cosine":
        T_max = params.get("T_max", 50)
        eta_min = params.get("eta_min", 0)
        return sched.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif name == "reduce_on_plateau":
        mode = params.get("mode", "min")
        factor = params.get("factor", 0.1)
        patience = params.get("patience", 10)
        return sched.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif name == "step":
        step_size = params.get("step_size", 10)
        gamma = params.get("gamma", 0.1)
        return sched.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "exponential":
        gamma = params.get("gamma", 0.95)
        return sched.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}")

