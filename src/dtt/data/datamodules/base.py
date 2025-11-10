from __future__ import annotations

from typing import Any

from dtt.utils.registry import register_datamodule


class _BaseLightningDataModule:
    """Minimal Lightning-like DataModule shell for type hints without hard dependency."""

    def __init__(self):
        pass

    # Placeholder methods matching LightningDataModule API
    def setup(self, stage: str | None = None):  # pragma: no cover
        pass

    def train_dataloader(self):  # pragma: no cover
        raise NotImplementedError

    def val_dataloader(self):  # pragma: no cover
        raise NotImplementedError


@register_datamodule("base")
def build_base_datamodule(cfg: dict[str, Any]):
    """Returns a pass-through base datamodule (not used directly)."""
    try:
        from lightning.pytorch import LightningDataModule  # type: ignore
    except Exception:  # pragma: no cover - optional dep

        class LightningDataModule(_BaseLightningDataModule): ...

    class BaseDataModule(LightningDataModule):
        pass

    return BaseDataModule()
