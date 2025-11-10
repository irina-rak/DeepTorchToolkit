from __future__ import annotations

from typing import Any, Dict

from dtt.config.schemas import Config
from dtt.models.base import build_optimizer, build_scheduler
from dtt.utils.registry import register_model


@register_model("monai.unet")
def build_monai_unet(cfg: Dict[str, Any]):
    """Factory that returns a LightningModule wrapping MONAI UNet.

    This import pattern keeps torch/monai heavy deps out of module import time.
    """
    from lightning.pytorch import LightningModule
    import torch

    try:
        from monai.networks.nets import UNet
        from monai.losses import DiceLoss
        from monai.metrics import DiceMetric
    except Exception:  # pragma: no cover - optional dependency
        UNet = None
        DiceLoss = None
        DiceMetric = None

    class MonaiUNetLightning(LightningModule):
        def __init__(self, config: Config | Dict[str, Any]):
            super().__init__()
            # normalize config
            if isinstance(config, dict):
                from dtt.config.schemas import Config as C

                config = C.model_validate(config)
            mcfg = config.model
            p = mcfg.params

            if UNet is None:
                raise ImportError("MONAI is not installed. Install with `pip install -e .[monai]`. ")

            self.model = UNet(
                spatial_dims=2,
                in_channels=int(p.get("in_channels", 1)),
                out_channels=int(p.get("out_channels", 1)),
                channels=p.get("channels", [16, 32, 64, 128]),
                strides=p.get("strides", [2, 2, 2]),
                num_res_units=int(p.get("num_res_units", 2)),
                norm=p.get("norm", "batch"),
            )
            loss_name = str(p.get("loss", "bce")).lower()
            if loss_name == "dice" and DiceLoss is not None:
                self.criterion = DiceLoss(sigmoid=True)
            else:
                self.criterion = torch.nn.BCEWithLogitsLoss()

            self.save_hyperparameters(ignore=["model", "criterion"])  # type: ignore[attr-defined]
            self.optim_cfg = mcfg.optim
            self.scheduler_cfg = mcfg.scheduler

            # Setup metrics if requested
            self.metrics_names = mcfg.metrics
            self.val_dice = None
            if "dice" in self.metrics_names and DiceMetric is not None:
                self.val_dice = DiceMetric(include_background=True, reduction="mean")

        def forward(self, x):  # type: ignore[override]
            return self.model(x)

        def training_step(self, batch, batch_idx):  # type: ignore[override]
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y.float())
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):  # type: ignore[override]
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y.float())
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

            # Compute metrics if configured
            if self.val_dice is not None:
                y_pred = torch.sigmoid(y_hat) > 0.5
                self.val_dice(y_pred=y_pred, y=y)

            return loss

        def on_validation_epoch_end(self):  # type: ignore[override]
            """Aggregate and log metrics at epoch end."""
            if self.val_dice is not None:
                dice_score = self.val_dice.aggregate()
                self.log("val/dice", dice_score, prog_bar=True)
                self.val_dice.reset()

        def configure_optimizers(self):  # type: ignore[override]
            optimizer = build_optimizer(self.parameters(), self.optim_cfg)

            # Return optimizer + scheduler if configured
            scheduler = build_scheduler(optimizer, self.scheduler_cfg)
            if scheduler is not None:
                # Check if ReduceLROnPlateau (needs monitor)
                scheduler_cfg_dict = {"scheduler": scheduler}
                if self.scheduler_cfg.name and "plateau" in self.scheduler_cfg.name.lower():
                    scheduler_cfg_dict["monitor"] = "val/loss"
                    scheduler_cfg_dict["interval"] = "epoch"
                return {"optimizer": optimizer, "lr_scheduler": scheduler_cfg_dict}

            return optimizer

    return MonaiUNetLightning(cfg)

