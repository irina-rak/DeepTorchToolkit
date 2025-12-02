"""Vector-Quantized Variational Autoencoder (VQ-VAE) with GAN training using MONAI.

This module implements a VQ-VAE for learning a discrete latent space,
which can be used as the first stage of a Latent Diffusion Model.
VQ-VAE typically produces sharper reconstructions than VAE due to the
discrete codebook.

GAN training (adversarial loss) further improves reconstruction quality
by producing sharper, more realistic outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from dtt.utils.registry import register_model

__all__ = ["build_vqvae"]


@register_model("vqvae")
def build_vqvae(cfg: dict[str, Any]):
    """Build VQ-VAE model from config.

    Expected config structure:
        model:
            name: "vqvae"
            optim:
                name: adam | adamw | sgd
                lr: float (e.g., 1e-4)
                weight_decay: float (default: 0.0)
            scheduler:
                name: cosine | reduce_on_plateau | step | exponential | null
                params: {}
            params:
                # VQVAE architecture
                spatial_dims: int (2 or 3)
                in_channels: int (default: 1)
                out_channels: int (default: 1)
                channels: list[int] (default: [96, 96, 192])
                num_res_layers: int (default: 3)
                num_res_channels: list[int] | int (default: [96, 96, 192])
                downsample_parameters: list[tuple] (default: encoder stride/kernel config)
                upsample_parameters: list[tuple] (default: decoder stride/kernel config)

                # Codebook settings
                num_embeddings: int (default: 256)
                    Number of codebook vectors
                embedding_dim: int (default: 64)
                    Dimension of each codebook vector
                commitment_cost: float (default: 0.25)
                    Weight for commitment loss (encoder to codebook)
                decay: float (default: 0.99)
                    EMA decay for codebook updates (0 = no EMA)

                # Loss weights
                perceptual_weight: float (default: 0.1)
                    Weight for perceptual loss
                adversarial_weight: float (default: 0.1)
                    Weight for adversarial (GAN) loss. Set to 0 to disable GAN training.
                reconstruction_loss: str (default: "l1")
                    Type of reconstruction loss: "l1", "l2", or "mse"

                # Discriminator architecture (only used if adversarial_weight > 0)
                discriminator_channels: int (default: 64)
                    Base channels for PatchGAN discriminator
                discriminator_num_layers: int (default: 3)
                    Number of discriminator layers

                # GAN training settings
                disc_start_epoch: int (default: 0)
                    Epoch to start discriminator training
                disc_loss_type: str (default: "hinge")
                    Discriminator loss type: "hinge", "vanilla", or "least_squares"

                # Training settings
                seed: int | None (default: None)
                _logging: bool (default: True)
    """
    import torch
    import torch.nn.functional as functional
    from lightning.pytorch import LightningModule
    from monai.networks.nets import VQVAE, PatchDiscriminator
    from monai.utils import set_determinism

    from dtt.models.base import build_optimizer, build_scheduler
    from dtt.utils.logging import get_console

    console = get_console()

    class LitVQVAE(LightningModule):
        """Lightning module for VQ-VAE with optional GAN training."""

        def __init__(self, config: Any):
            super().__init__()

            # Normalize config to Config object
            if isinstance(config, dict):
                from dtt.config.schemas import Config as ConfigSchema

                config = ConfigSchema.model_validate(config)

            mcfg = config.model
            p = mcfg.params

            # Disable automatic optimization for GAN training
            self.automatic_optimization = False

            # Set seed if provided
            seed = p.get("seed")
            if seed is not None:
                set_determinism(seed=seed)
                console.log(f"[bold][yellow]Setting seed to {seed}.[/yellow][/bold]")

            # Extract architecture config
            spatial_dims = p.get("spatial_dims", 2)
            in_channels = p.get("in_channels", 1)
            out_channels = p.get("out_channels", 1)
            channels = p.get("channels", [96, 96, 192])
            num_res_layers = p.get("num_res_layers", 3)
            num_res_channels = p.get("num_res_channels", channels)

            # Codebook settings
            num_embeddings = p.get("num_embeddings", 256)
            embedding_dim = p.get("embedding_dim", 64)
            commitment_cost = p.get("commitment_cost", 0.25)
            decay = p.get("decay", 0.99)

            # Downsample/upsample parameters
            num_levels = len(channels)
            default_downsample = tuple((2, 4, 1, 1) for _ in range(num_levels))
            default_upsample = tuple((2, 4, 1, 1, 0) for _ in range(num_levels))

            downsample_parameters = p.get("downsample_parameters", default_downsample)
            upsample_parameters = p.get("upsample_parameters", default_upsample)

            # Initialize VQVAE
            self.model = VQVAE(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                num_res_layers=num_res_layers,
                num_res_channels=num_res_channels,
                downsample_parameters=downsample_parameters,
                upsample_parameters=upsample_parameters,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                ddp_sync=True,  # Sync codebook across GPUs
            )

            # Loss configuration
            self.perceptual_weight = p.get("perceptual_weight", 0.1)
            self.adversarial_weight = p.get("adversarial_weight", 0.1)
            self.reconstruction_loss_type = p.get("reconstruction_loss", "l1").lower()

            # GAN settings
            self.disc_start_epoch = p.get("disc_start_epoch", 0)
            self.disc_loss_type = p.get("disc_loss_type", "hinge").lower()

            # Initialize discriminator if GAN training enabled
            self.discriminator = None
            if self.adversarial_weight > 0:
                disc_channels = p.get("discriminator_channels", 64)
                disc_num_layers = p.get("discriminator_num_layers", 3)

                self.discriminator = PatchDiscriminator(
                    spatial_dims=spatial_dims,
                    channels=disc_channels,
                    in_channels=out_channels,
                    out_channels=1,
                    num_layers_d=disc_num_layers,
                    kernel_size=4,
                    activation=("LEAKYRELU", {"negative_slope": 0.2}),
                    norm="BATCH",
                    bias=False,
                    padding=1,
                )
                console.log(
                    f"[cyan]GAN training enabled with adversarial_weight={self.adversarial_weight}, "
                    f"disc_start_epoch={self.disc_start_epoch}[/cyan]"
                )

            # Initialize perceptual loss if needed
            self.perceptual_loss = None
            if self.perceptual_weight > 0:
                try:
                    from monai.losses import PerceptualLoss

                    self.perceptual_loss = PerceptualLoss(
                        spatial_dims=spatial_dims,
                        network_type="alex",
                        is_fake_3d=True if spatial_dims == 3 else False,
                        fake_3d_ratio=0.5,
                    )
                    console.log(
                        f"[cyan]Perceptual loss enabled with weight={self.perceptual_weight}[/cyan]"
                    )
                except Exception as e:
                    console.log(
                        f"[yellow]Warning: PerceptualLoss not available ({e}), "
                        f"disabling perceptual loss[/yellow]"
                    )
                    self.perceptual_weight = 0.0

            # Logging settings
            self._logging = p.get("_logging", True)

            # Store optimizer and scheduler configs
            self.optim_cfg = mcfg.optim
            self.scheduler_cfg = mcfg.scheduler

            # Store spatial dims and embedding info
            self.spatial_dims = spatial_dims
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim

            self.save_hyperparameters(ignore=["model", "discriminator", "perceptual_loss"])

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Encode input to quantized latent representation."""
            return self.model.encode(x)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Decode quantized latent to reconstruction."""
            return self.model.decode(z)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass: encode, quantize, and decode."""
            recon, quantization_loss = self.model(x)
            return recon, quantization_loss

        def _compute_reconstruction_loss(
            self, recon: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            """Compute reconstruction loss."""
            if self.reconstruction_loss_type == "l1":
                return functional.l1_loss(recon, target)
            elif self.reconstruction_loss_type in ("l2", "mse"):
                return functional.mse_loss(recon, target)
            else:
                raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss_type}")

        def _compute_generator_adversarial_loss(
            self, fake_logits: torch.Tensor | list[torch.Tensor]
        ) -> torch.Tensor:
            """Compute generator adversarial loss.

            PatchDiscriminator returns a list of feature maps, so we average over all scales.
            """
            # Handle list output from PatchDiscriminator (multi-scale)
            if isinstance(fake_logits, list):
                losses = [self._compute_generator_adversarial_loss(fl) for fl in fake_logits]
                return torch.stack(losses).mean()

            if self.disc_loss_type == "hinge":
                return -torch.mean(fake_logits)
            elif self.disc_loss_type == "vanilla":
                return functional.binary_cross_entropy_with_logits(
                    fake_logits, torch.ones_like(fake_logits)
                )
            elif self.disc_loss_type == "least_squares":
                return 0.5 * torch.mean((fake_logits - 1) ** 2)
            else:
                raise ValueError(f"Unknown disc_loss_type: {self.disc_loss_type}")

        def _compute_discriminator_loss(
            self,
            real_logits: torch.Tensor | list[torch.Tensor],
            fake_logits: torch.Tensor | list[torch.Tensor],
        ) -> torch.Tensor:
            """Compute discriminator loss.

            PatchDiscriminator returns a list of feature maps, so we average over all scales.
            """
            # Handle list output from PatchDiscriminator (multi-scale)
            if isinstance(real_logits, list) and isinstance(fake_logits, list):
                losses = [
                    self._compute_discriminator_loss(rl, fl)
                    for rl, fl in zip(real_logits, fake_logits, strict=True)
                ]
                return torch.stack(losses).mean()

            if self.disc_loss_type == "hinge":
                real_loss = torch.mean(functional.relu(1.0 - real_logits))
                fake_loss = torch.mean(functional.relu(1.0 + fake_logits))
                return 0.5 * (real_loss + fake_loss)
            elif self.disc_loss_type == "vanilla":
                real_loss = functional.binary_cross_entropy_with_logits(
                    real_logits, torch.ones_like(real_logits)
                )
                fake_loss = functional.binary_cross_entropy_with_logits(
                    fake_logits, torch.zeros_like(fake_logits)
                )
                return 0.5 * (real_loss + fake_loss)
            elif self.disc_loss_type == "least_squares":
                real_loss = 0.5 * torch.mean((real_logits - 1) ** 2)
                fake_loss = 0.5 * torch.mean(fake_logits**2)
                return 0.5 * (real_loss + fake_loss)
            else:
                raise ValueError(f"Unknown disc_loss_type: {self.disc_loss_type}")

        def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            """Training step with GAN training."""
            images = batch["image"]

            # Get optimizers
            if self.discriminator is not None:
                opt_g, opt_d = self.optimizers()
            else:
                opt_g = self.optimizers()
                opt_d = None

            # Check if discriminator should be active
            disc_active = (
                self.discriminator is not None and self.current_epoch >= self.disc_start_epoch
            )

            # ==================== Generator (VQ-VAE) Update ====================
            opt_g.zero_grad()

            # Forward pass
            recon, quantization_loss = self(images)

            # Reconstruction loss
            recon_loss = self._compute_reconstruction_loss(recon, images)

            # Generator loss (includes VQ commitment loss)
            g_loss = recon_loss + quantization_loss

            # Perceptual loss
            perceptual_loss = torch.tensor(0.0, device=self.device)
            if self.perceptual_loss is not None and self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_loss(recon, images)
                g_loss = g_loss + self.perceptual_weight * perceptual_loss

            # Adversarial loss for generator
            g_adv_loss = torch.tensor(0.0, device=self.device)
            if disc_active:
                fake_logits = self.discriminator(recon)
                g_adv_loss = self._compute_generator_adversarial_loss(fake_logits)
                g_loss = g_loss + self.adversarial_weight * g_adv_loss

            # Backward and update generator
            self.manual_backward(g_loss)
            opt_g.step()

            # ==================== Discriminator Update ====================
            d_loss = torch.tensor(0.0, device=self.device)
            if disc_active and opt_d is not None:
                opt_d.zero_grad()

                # Get discriminator predictions
                with torch.no_grad():
                    recon_detached = recon.detach()

                real_logits = self.discriminator(images)
                fake_logits = self.discriminator(recon_detached)

                # Discriminator loss
                d_loss = self._compute_discriminator_loss(real_logits, fake_logits)

                # Backward and update discriminator
                self.manual_backward(d_loss)
                opt_d.step()

            # Logging
            if self._logging:
                self.log("train/loss", g_loss, prog_bar=True, sync_dist=True)
                self.log("train/recon_loss", recon_loss, prog_bar=False, sync_dist=True)
                self.log("train/vq_loss", quantization_loss, prog_bar=False, sync_dist=True)
                if self.perceptual_weight > 0:
                    self.log(
                        "train/perceptual_loss", perceptual_loss, prog_bar=False, sync_dist=True
                    )
                if disc_active:
                    self.log("train/g_adv_loss", g_adv_loss, prog_bar=False, sync_dist=True)
                    self.log("train/d_loss", d_loss, prog_bar=True, sync_dist=True)

                # Log codebook usage metrics periodically
                if batch_idx % 100 == 0:
                    self._log_codebook_metrics()

            return {"loss": g_loss}

        def validation_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            """Validation step."""
            images = batch["image"]

            # Forward pass
            recon, quantization_loss = self(images)

            # Compute losses
            recon_loss = self._compute_reconstruction_loss(recon, images)
            loss = recon_loss + quantization_loss

            if self.perceptual_loss is not None and self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_loss(recon, images)
                loss = loss + self.perceptual_weight * perceptual_loss

            if self._logging:
                self.log("val/loss", loss, prog_bar=True, sync_dist=True)
                self.log("val/recon_loss", recon_loss, prog_bar=False, sync_dist=True)
                self.log("val/vq_loss", quantization_loss, prog_bar=False, sync_dist=True)

            return {"loss": loss}

        def _log_codebook_metrics(self):
            """Log codebook usage metrics."""
            try:
                if hasattr(self.model, "quantizer"):
                    quantizer = self.model.quantizer
                    if hasattr(quantizer, "embedding"):
                        embedding = quantizer.embedding
                        self.log(
                            "codebook/embedding_norm",
                            embedding.weight.norm(dim=1).mean(),
                            sync_dist=True,
                        )
            except Exception:
                pass

        def on_validation_epoch_end(self) -> None:
            """Save reconstruction samples for visualization."""
            if not self.trainer.is_global_zero:
                return

            self._save_reconstruction_samples()

        def _save_reconstruction_samples(self, max_samples: int = 4):
            """Save reconstruction samples for visual monitoring."""
            import os

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            val_loader = self.trainer.datamodule.val_dataloader()
            batch = next(iter(val_loader))
            images = batch["image"].to(self.device)

            # Limit samples
            images = images[:max_samples]

            with torch.no_grad():
                recon, _ = self(images)

            # Get log directory
            log_dir = self._get_log_dir()
            samples_dir = os.path.join(log_dir, "vqvae_samples")
            os.makedirs(samples_dir, exist_ok=True)

            # Save comparison images
            for i in range(min(max_samples, images.shape[0])):
                orig = images[i].cpu().squeeze()
                rec = recon[i].cpu().squeeze()

                # Handle 3D: take center slice
                if orig.dim() == 3:
                    center = orig.shape[0] // 2
                    orig = orig[center]
                    rec = rec[center]

                # Normalize for display
                orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
                rec = (rec - rec.min()) / (rec.max() - rec.min() + 1e-8)

                # Rotate for correct orientation
                orig = np.rot90(orig.numpy(), k=-1)
                rec = np.rot90(rec.numpy(), k=-1)

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(orig, cmap="gray")
                axes[0].set_title("Original")
                axes[0].axis("off")

                axes[1].imshow(rec, cmap="gray")
                axes[1].set_title("Reconstruction")
                axes[1].axis("off")

                plt.tight_layout()
                save_path = os.path.join(
                    samples_dir, f"epoch_{self.current_epoch:04d}_sample_{i}.png"
                )
                plt.savefig(save_path, bbox_inches="tight", dpi=100)
                plt.close(fig)

            console.log(f"[dim]Saved VQ-VAE reconstruction samples to: {samples_dir}[/dim]")

        def _get_log_dir(self) -> str:
            """Get the logging directory."""
            import os

            log_dir = None

            if self.trainer.logger is not None:
                if hasattr(self.trainer.logger, "experiment"):
                    if hasattr(self.trainer.logger.experiment, "dir"):
                        log_dir = self.trainer.logger.experiment.dir
                    elif hasattr(self.trainer.logger.experiment, "path"):
                        log_dir = str(self.trainer.logger.experiment.path)

                if not log_dir:
                    if hasattr(self.trainer.logger, "save_dir") and self.trainer.logger.save_dir:
                        log_dir = self.trainer.logger.save_dir
                    elif hasattr(self.trainer.logger, "log_dir") and self.trainer.logger.log_dir:
                        log_dir = self.trainer.logger.log_dir

            if not log_dir or log_dir == ".":
                log_dir = os.path.abspath(self.trainer.default_root_dir)

            return log_dir

        def on_train_epoch_end(self) -> None:
            """Calculate scaling factor at the end of the last epoch.

            This ensures the scaling factor is computed BEFORE the final checkpoint
            is saved (on_train_end runs after checkpointing).
            """
            # Only compute on the last epoch and on rank 0
            if not self.trainer.is_global_zero:
                return

            is_last_epoch = self.current_epoch == self.trainer.max_epochs - 1
            if not is_last_epoch:
                return

            console.print("[bold cyan]Calculating scaling factor for downstream LDM...[/bold cyan]")

            latent_stds = []
            with torch.no_grad():
                for i, batch in enumerate(self.trainer.datamodule.train_dataloader()):
                    images = batch["image"].to(self.device)
                    z = self.encode(images)
                    latent_stds.append(torch.std(z).item())

                    if i >= 100:
                        break

            if latent_stds:
                avg_std = sum(latent_stds) / len(latent_stds)
                scaling_factor = 1.0 / (avg_std + 1e-8)
                scaling_factor = max(0.01, min(10.0, scaling_factor))

                # Store as buffer so it gets saved in checkpoint
                self.register_buffer(
                    "scaling_factor", torch.tensor(scaling_factor, device=self.device)
                )

                console.print(
                    f"[bold green]Calculated scale_factor for LDM: {scaling_factor:.4f}[/bold green]"
                )
                console.print(f"[dim]Latent std: {avg_std:.4f} (scale_factor = 1/std)[/dim]")

        def on_save_checkpoint(self, checkpoint: dict) -> None:
            """Save scaling factor to checkpoint if available."""
            if hasattr(self, "scaling_factor"):
                checkpoint["scaling_factor"] = self.scaling_factor.cpu()
                console.print(
                    f"[dim]Saved scaling_factor={self.scaling_factor.item():.4f} to checkpoint[/dim]"
                )

        def configure_optimizers(self):
            """Configure optimizer and scheduler for GAN training."""
            # Generator optimizer
            opt_g = build_optimizer(self.model.parameters(), self.optim_cfg)

            optimizers = [opt_g]
            schedulers = []

            # Generator scheduler
            scheduler_g = build_scheduler(opt_g, self.scheduler_cfg)
            if scheduler_g is not None:
                scheduler_cfg_dict = {
                    "scheduler": scheduler_g,
                    "interval": "epoch",
                    "frequency": 1,
                }
                if self.scheduler_cfg.name and "plateau" in self.scheduler_cfg.name.lower():
                    scheduler_cfg_dict["monitor"] = "val/loss"
                schedulers.append(scheduler_cfg_dict)

            # Discriminator optimizer (if GAN training enabled)
            if self.discriminator is not None:
                opt_d = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=self.optim_cfg.lr,
                    weight_decay=self.optim_cfg.weight_decay,
                    betas=(0.5, 0.9),  # Standard GAN betas
                )
                optimizers.append(opt_d)

                # Discriminator scheduler
                scheduler_d = build_scheduler(opt_d, self.scheduler_cfg)
                if scheduler_d is not None:
                    scheduler_cfg_dict_d = {
                        "scheduler": scheduler_d,
                        "interval": "epoch",
                        "frequency": 1,
                    }
                    if self.scheduler_cfg.name and "plateau" in self.scheduler_cfg.name.lower():
                        scheduler_cfg_dict_d["monitor"] = "val/loss"
                    schedulers.append(scheduler_cfg_dict_d)

            if schedulers:
                return optimizers, schedulers
            return optimizers

    return LitVQVAE(cfg)
