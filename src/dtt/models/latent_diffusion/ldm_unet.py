"""Latent Diffusion Model U-Net using MONAI's DiffusionModelUNet.

This module implements a diffusion model that operates in the latent space
of a pre-trained autoencoder (VAE or VQ-VAE). This is the second stage of
training a Latent Diffusion Model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from dtt.utils.registry import register_model

__all__ = ["build_ldm_unet"]


@register_model("ldm_unet")
def build_ldm_unet(cfg: dict[str, Any]):
    """Build Latent Diffusion Model U-Net from config.

    Expected config structure:
        model:
            name: "ldm_unet"
            optim:
                name: adam | adamw | sgd
                lr: float (e.g., 1e-4)
                weight_decay: float (default: 0.0)
            scheduler:
                name: cosine | reduce_on_plateau | step | exponential | null
                params: {}
            params:
                # Pre-trained autoencoder (REQUIRED)
                autoencoder_checkpoint: str
                    Path to the pre-trained VAE or VQ-VAE checkpoint
                autoencoder_type: str (default: "vae")
                    Type of autoencoder: "vae" or "vqvae"
                autoencoder_config: dict
                    Configuration for the autoencoder (same as used during training)
                scaling_factor: float (default: 1.0)
                    Scaling factor for latent space (set based on autoencoder's latent variance)

                # DiffusionModelUNet architecture
                spatial_dims: int (2 or 3)
                in_channels: int (must match autoencoder's latent_channels)
                out_channels: int (must match autoencoder's latent_channels)
                num_res_blocks: list[int] | int (default: [2, 2, 2, 2])
                channels: list[int] (default: [128, 256, 512, 1024])
                attention_levels: list[bool] (default: [False, True, True, True])
                norm_num_groups: int (default: 32)
                num_head_channels: int | list[int] (default: 64)
                resblock_updown: bool (default: False)
                use_flash_attention: bool (default: False)

                # Conditioning (optional)
                with_conditioning: bool (default: False)
                cross_attention_dim: int | None (default: None)
                num_class_embeds: int | None (default: None)

                # Diffusion settings
                num_train_timesteps: int (default: 1000)
                beta_start: float (default: 0.0001)
                beta_end: float (default: 0.02)
                beta_schedule: str (default: "linear")
                    Schedule type: "linear", "cosine", or "scaled_linear"
                prediction_type: str (default: "epsilon")
                    What the model predicts: "epsilon" (noise) or "v_prediction"

                # Training settings
                manual_accumulate_grad_batches: int (default: 4)
                seed: int | None (default: None)
                _logging: bool (default: True)

                # EMA settings
                use_ema: bool (default: False)
                ema_decay: float (default: 0.9999)

                # Validation generation settings
                generate_validation_samples: bool (default: True)
                generate_frequency: int (default: 5)
                val_max_batches: int (default: 3)
                num_inference_steps: int (default: 50)
    """
    import os

    import torch
    import torch.nn.functional as functional
    from lightning.pytorch import LightningModule
    from monai.inferers import LatentDiffusionInferer
    from monai.networks.nets import VQVAE, AutoencoderKL, DiffusionModelUNet
    from monai.networks.schedulers import DDPMScheduler
    from monai.utils import first, set_determinism

    from dtt.models.base import build_optimizer, build_scheduler
    from dtt.utils.logging import get_console

    console = get_console()

    class LitLDMUNet(LightningModule):
        """Lightning module for Latent Diffusion Model U-Net."""

        def __init__(self, config: Any):
            super().__init__()

            # Normalize config to Config object
            if isinstance(config, dict):
                from dtt.config.schemas import Config as ConfigSchema

                config = ConfigSchema.model_validate(config)

            mcfg = config.model
            p = mcfg.params

            # Disable automatic optimization for manual gradient accumulation
            self.automatic_optimization = False

            # Set seed if provided
            seed = p.get("seed")
            if seed is not None:
                set_determinism(seed=seed)
                console.log(f"[bold][yellow]Setting seed to {seed}.[/yellow][/bold]")

            # Load pre-trained autoencoder
            self.autoencoder_type = p.get("autoencoder_type", "vae").lower()
            self.autoencoder_checkpoint = p.get("autoencoder_checkpoint")
            self.autoencoder_config = p.get("autoencoder_config", {})
            # self.scaling_factor = p.get("scaling_factor", 1.0)

            if self.autoencoder_checkpoint is None:
                raise ValueError(
                    "autoencoder_checkpoint is required. Please provide the path to a "
                    "pre-trained VAE or VQ-VAE checkpoint."
                )

            # Initialize and load autoencoder
            self.autoencoder = self._load_autoencoder()
            self.autoencoder.eval()
            self.autoencoder.requires_grad_(False)
            console.log(
                f"[bold green]Loaded {self.autoencoder_type.upper()} from: "
                f"{self.autoencoder_checkpoint}[/bold green]"
            )

            # Extract U-Net architecture config
            spatial_dims = p.get("spatial_dims", 2)
            in_channels = p.get("in_channels", 4)  # Typically latent_channels from autoencoder
            out_channels = p.get("out_channels", 4)
            num_res_blocks = p.get("num_res_blocks", [2, 2, 2, 2])
            channels = p.get("channels", [128, 256, 512, 1024])
            attention_levels = p.get("attention_levels", [False, True, True, True])
            norm_num_groups = p.get("norm_num_groups", 32)
            num_head_channels = p.get("num_head_channels", 64)

            # Conditioning settings
            with_conditioning = p.get("with_conditioning", False)
            cross_attention_dim = p.get("cross_attention_dim", None)
            num_class_embeds = p.get("num_class_embeds", None)

            # Initialize DiffusionModelUNet
            self.unet = DiffusionModelUNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                num_res_blocks=num_res_blocks,
                channels=channels,
                attention_levels=attention_levels,
                norm_num_groups=norm_num_groups,
                num_head_channels=num_head_channels,
                resblock_updown=p.get("resblock_updown", False),
                with_conditioning=with_conditioning,
                cross_attention_dim=cross_attention_dim,
                num_class_embeds=num_class_embeds,
                use_flash_attention=p.get("use_flash_attention", False),
            )

            # Initialize noise scheduler
            num_train_timesteps = p.get("num_train_timesteps", 1000)
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=p.get("beta_start", 0.0015),
                beta_end=p.get("beta_end", 0.0195),
                schedule=p.get("beta_schedule", "linear_beta"),
                prediction_type=p.get("prediction_type", "epsilon"),
            )

            # LatentDiffusionInferer for training and sampling
            # If scaling_factor is None (will be computed in on_fit_start), use 1.0 temporarily
            if self.scaling_factor is not None:
                scale_factor_value = (
                    self.scaling_factor.item()
                    if isinstance(self.scaling_factor, torch.Tensor)
                    else self.scaling_factor
                )
            else:
                scale_factor_value = 1.0  # Temporary, will be updated in on_fit_start
            self.inferer = LatentDiffusionInferer(
                scheduler=self.scheduler,
                scale_factor=scale_factor_value,
            )

            # Store settings
            self.num_train_timesteps = num_train_timesteps
            self.prediction_type = p.get("prediction_type", "epsilon")
            self.manual_accumulate_grad_batches = p.get("manual_accumulate_grad_batches", 4)
            self._logging = p.get("_logging", True)
            self.spatial_dims = spatial_dims

            # Validation generation settings
            self.generate_validation_samples = p.get("generate_validation_samples", True)
            self.generate_frequency = p.get("generate_frequency", 5)
            self.val_max_batches = p.get("val_max_batches", 3)
            self.num_inference_steps = p.get("num_inference_steps", 50)

            # EMA settings
            self.use_ema = p.get("use_ema", False)
            self.ema_decay = p.get("ema_decay", 0.9999)
            if self.use_ema:
                self.unet_ema = self._create_ema_model()
                console.log(f"[bold cyan]EMA enabled with decay={self.ema_decay}[/bold cyan]")

            # Store optimizer and scheduler configs
            self.optim_cfg = mcfg.optim
            self.scheduler_cfg = mcfg.scheduler

            self.save_hyperparameters(ignore=["autoencoder", "unet", "unet_ema", "scheduler"])

        def _load_autoencoder(self):
            """Load pre-trained autoencoder from checkpoint."""
            ae_cfg = self.autoencoder_config

            if self.autoencoder_type == "vae":
                autoencoder = AutoencoderKL(
                    spatial_dims=ae_cfg.get("spatial_dims", 2),
                    in_channels=ae_cfg.get("in_channels", 1),
                    out_channels=ae_cfg.get("out_channels", 1),
                    channels=ae_cfg.get("channels", [64, 128, 256, 512]),
                    num_res_blocks=ae_cfg.get("num_res_blocks", 2),
                    latent_channels=ae_cfg.get("latent_channels", 4),
                    attention_levels=ae_cfg.get("attention_levels", [False, False, True, True]),
                    norm_num_groups=ae_cfg.get("norm_num_groups", 32),
                    with_encoder_nonlocal_attn=ae_cfg.get("with_encoder_nonlocal_attn", True),
                    with_decoder_nonlocal_attn=ae_cfg.get("with_decoder_nonlocal_attn", True),
                )
            elif self.autoencoder_type == "vqvae":
                autoencoder = VQVAE(
                    spatial_dims=ae_cfg.get("spatial_dims", 2),
                    in_channels=ae_cfg.get("in_channels", 1),
                    out_channels=ae_cfg.get("out_channels", 1),
                    channels=ae_cfg.get("channels", [96, 96, 192]),
                    num_res_layers=ae_cfg.get("num_res_layers", 3),
                    num_res_channels=ae_cfg.get("num_res_channels", [96, 96, 192]),
                    num_embeddings=ae_cfg.get("num_embeddings", 256),
                    embedding_dim=ae_cfg.get("embedding_dim", 64),
                )
            else:
                raise ValueError(
                    f"Unknown autoencoder type: {self.autoencoder_type}. "
                    "Supported types: 'vae', 'vqvae'"
                )

            # Load checkpoint
            if os.path.exists(self.autoencoder_checkpoint):
                # Note: weights_only=False needed for Lightning checkpoints that may contain
                # numpy arrays or other non-tensor objects. Only load from trusted sources.
                checkpoint = torch.load(
                    self.autoencoder_checkpoint, map_location="cpu", weights_only=False
                )

                # Handle Lightning checkpoint format
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    # Remove "model." prefix if present (from LightningModule)
                    state_dict = {
                        k.replace("model.", ""): v
                        for k, v in state_dict.items()
                        if k.startswith("model.")
                    }
                else:
                    state_dict = checkpoint

                autoencoder.load_state_dict(state_dict, strict=True)

                # Load scaling factor if available in checkpoint
                if "scaling_factor" in checkpoint:
                    self.scaling_factor = checkpoint["scaling_factor"]
                    console.log(
                        f"[bold green]Loaded scaling factor from checkpoint: "
                        f"{self.scaling_factor}[/bold green]"
                    )
                else:
                    # Defer scaling factor computation to on_fit_start when trainer is available
                    self.scaling_factor = None
                    console.log(
                        "[bold yellow]Scaling factor not found in checkpoint. "
                        "Will be computed from training data in on_fit_start.[/bold yellow]"
                    )

            else:
                raise FileNotFoundError(
                    f"Autoencoder checkpoint not found: {self.autoencoder_checkpoint}"
                )

            return autoencoder

        @torch.no_grad()
        def _compute_scaling_factor(self):
            """Compute scaling factor based on latent space standard deviation
            of the first batch from the training dataloader."""
            console.log("[bold][cyan]Computing scaling factor from training data...[/cyan][/bold]")

            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                train_loader = self.trainer.datamodule.train_dataloader()
                batch = first(train_loader)
                images = batch["image"].to(self.device)

                # Encode to latent space
                z = self.autoencoder.encode_stage_2_inputs(images)

                # Compute std dev of latent representations
                latent_std = torch.std(z)

                # Scaling factor is inverse of std dev
                scaling_factor = 1 / (latent_std + 1e-8)

                console.log(
                    f"[bold][blue]Computed scaling factor: {scaling_factor.item():.4f}[/blue][/bold]"
                )

            return scaling_factor

        def _create_ema_model(self):
            """Create EMA copy of the U-Net."""
            import copy

            ema_model = copy.deepcopy(self.unet)
            ema_model.eval()
            ema_model.requires_grad_(False)
            return ema_model

        def _update_ema(self):
            """Update EMA model with exponential moving average."""
            if not self.use_ema:
                return

            with torch.no_grad():
                for ema_param, model_param in zip(
                    self.unet_ema.parameters(), self.unet.parameters(), strict=True
                ):
                    ema_param.data.mul_(self.ema_decay).add_(
                        model_param.data, alpha=1 - self.ema_decay
                    )

        def on_fit_start(self) -> None:
            """Compute scaling factor at start of training if not loaded from checkpoint."""
            # Compute scaling factor if not already loaded
            if self.scaling_factor is None:
                self.scaling_factor = self._compute_scaling_factor()

            # Create/update LatentDiffusionInferer with correct scaling factor
            scale_factor_value = (
                self.scaling_factor.item()
                if isinstance(self.scaling_factor, torch.Tensor)
                else self.scaling_factor
            )
            self.inferer = LatentDiffusionInferer(
                scheduler=self.scheduler,
                scale_factor=scale_factor_value,
            )
            console.log(
                f"[bold green]Created LatentDiffusionInferer with scale_factor={scale_factor_value:.4f}[/bold green]"
            )

        @torch.no_grad()
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Encode image to latent space using frozen autoencoder.

            Note: This method does NOT apply scaling_factor. The LatentDiffusionInferer
            handles scaling internally. This is primarily used for getting latent shapes
            or for manual encoding when not using the inferer.

            Args:
                x: Input image tensor (B, C, *spatial_dims)

            Returns:
                Latent tensor (unscaled)
            """
            if self.autoencoder_type == "vae":
                # Note: MONAI's encode() returns (z_mu, z_sigma) where z_sigma is std dev, not log variance
                z_mu, z_sigma = self.autoencoder.encode(x)
                z = self.autoencoder.sampling(z_mu, z_sigma)
            else:  # vqvae
                z = self.autoencoder.encode(x)

            return z

        @torch.no_grad()
        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Decode latent to image using frozen autoencoder.

            Note: This method does NOT apply inverse scaling_factor. The LatentDiffusionInferer
            handles scaling internally in its sample() method.

            Args:
                z: Latent tensor (unscaled)

            Returns:
                Decoded image tensor
            """
            return self.autoencoder.decode(z)

        def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            context: torch.Tensor | None = None,
            class_labels: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward pass through U-Net.

            Args:
                x: Noisy latent tensor
                timesteps: Diffusion timesteps
                context: Optional cross-attention conditioning
                class_labels: Optional class labels for conditioning

            Returns:
                Model prediction (noise or v-prediction)
            """
            return self.unet(
                x=x,
                timesteps=timesteps,
                context=context,
                class_labels=class_labels,
            )

        def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            """Training step with manual gradient accumulation.

            Uses LatentDiffusionInferer for encoding and noise prediction.
            """
            images = batch["image"]

            # Get optimizer
            opt = self.optimizers()

            # Determine if we should step optimizer
            is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
            should_step_optimizer = (
                (batch_idx + 1) % self.manual_accumulate_grad_batches == 0
            ) or is_last_batch

            if should_step_optimizer:
                opt.zero_grad(set_to_none=True)

            # Get latent shape for noise sampling
            with torch.no_grad():
                z = self.encode(images)
            noise = torch.randn_like(z)
            timesteps = torch.randint(
                0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=self.device
            ).long()

            # # Get conditioning if available
            # context = batch.get("context")
            # class_labels = batch.get("class_labels")

            # Use inferer - it handles: encode -> scale -> add_noise -> predict
            # Note: inferer doesn't support class_labels, only cross-attention condition
            noise_pred = self.inferer(
                inputs=images,
                diffusion_model=self.unet,
                noise=noise,
                timesteps=timesteps,
                autoencoder_model=self.autoencoder,
                # condition=context,
            )

            loss = functional.mse_loss(noise_pred, noise)

            # Scale loss for gradient accumulation
            loss_scaled = loss / self.manual_accumulate_grad_batches

            # Manual backward
            self.manual_backward(loss_scaled)

            # Step optimizer if accumulated enough gradients
            if should_step_optimizer:
                opt.step()
                if self.use_ema:
                    self._update_ema()

            if self._logging:
                self.log("train/loss", loss, prog_bar=True, sync_dist=True)

            return {"loss": loss}

        def validation_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            """Validation step using inferer."""
            images = batch["image"]

            # Get latent shape for noise sampling
            with torch.no_grad():
                z = self.encode(images)
            noise = torch.randn_like(z)
            timesteps = torch.randint(
                0, self.num_train_timesteps, (z.shape[0],), device=self.device
            ).long()

            # Get conditioning
            context = batch.get("context")

            # Use EMA model for validation if available
            unet_to_use = self.unet_ema if self.use_ema else self.unet

            # Use inferer for encoding and prediction
            noise_pred = self.inferer(
                inputs=images,
                autoencoder_model=self.autoencoder,
                diffusion_model=unet_to_use,
                noise=noise,
                timesteps=timesteps,
                condition=context,
            )

            loss = functional.mse_loss(noise_pred, noise)

            if self._logging:
                self.log("val/loss", loss, prog_bar=True, sync_dist=True)

            return {"loss": loss}

        def on_validation_epoch_end(self) -> None:
            """Generate samples at the end of validation epoch."""
            if not self.trainer.is_global_zero:
                return

            should_generate = (
                self.generate_validation_samples
                and self.current_epoch % self.generate_frequency == 0
            )

            if should_generate:
                self._generate_samples()

        @torch.no_grad()
        def _generate_samples(self, num_samples: int = 1):
            """Generate samples using LatentDiffusionInferer."""
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            console.log(f"[cyan]Generating {num_samples} samples...[/cyan]")

            # Use EMA model if available
            unet_to_use = self.unet_ema if self.use_ema else self.unet
            unet_to_use.eval()

            # Get latent shape from a validation batch
            val_loader = self.trainer.datamodule.val_dataloader()
            batch = next(iter(val_loader))
            images = batch["image"].to(self.device)[:num_samples]
            z = self.encode(images)
            latent_shape = z.shape

            # Start from random noise in latent space
            noise = torch.randn(num_samples, *latent_shape[1:], device=self.device)

            # Set inference timesteps
            self.scheduler.set_timesteps(self.num_inference_steps)

            # Use LatentDiffusionInferer for sampling (handles decoding internally)
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                generated_images = self.inferer.sample(
                    input_noise=noise,
                    autoencoder_model=self.autoencoder,
                    diffusion_model=unet_to_use,
                    scheduler=self.scheduler,
                    verbose=False,
                )

            # Save samples
            log_dir = self._get_log_dir()
            import os

            samples_dir = os.path.join(log_dir, "ldm_samples")
            os.makedirs(samples_dir, exist_ok=True)

            wandb_images = []
            for i in range(num_samples):
                img = generated_images[i].cpu().squeeze()

                # Handle 3D: take center slice
                if img.dim() == 3:
                    center = img.shape[0] // 2
                    img = img[center]

                # Normalize for display
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = np.rot90(img.numpy(), k=-1)

                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(img, cmap="gray")
                ax.set_title(f"Generated (Epoch {self.current_epoch})")
                ax.axis("off")

                plt.tight_layout()
                save_path = os.path.join(
                    samples_dir, f"epoch_{self.current_epoch:04d}_sample_{i}.png"
                )
                plt.savefig(save_path, bbox_inches="tight", dpi=100)
                plt.close(fig)

                # Collect for W&B logging
                wandb_images.append(img)

            # Log images to W&B if available
            if self.logger is not None and hasattr(self.logger, "experiment"):
                try:
                    import wandb

                    # Log as a grid of images
                    self.logger.experiment.log(
                        {
                            "generated_samples": [
                                wandb.Image(img, caption=f"Sample {i}")
                                for i, img in enumerate(wandb_images)
                            ],
                            "epoch": self.current_epoch,
                        }
                    )
                except Exception as e:
                    console.log(f"[yellow]Could not log images to W&B: {e}[/yellow]")

            console.log(f"[dim]Saved LDM samples to: {samples_dir}[/dim]")

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

        def configure_optimizers(self):
            """Configure optimizer and scheduler."""
            # Only optimize U-Net parameters (autoencoder is frozen)
            optimizer = build_optimizer(self.unet.parameters(), self.optim_cfg)

            scheduler = build_scheduler(optimizer, self.scheduler_cfg)
            if scheduler is not None:
                scheduler_cfg_dict = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }

                if self.scheduler_cfg.name and "plateau" in self.scheduler_cfg.name.lower():
                    scheduler_cfg_dict["monitor"] = "val/loss"

                return {"optimizer": optimizer, "lr_scheduler": scheduler_cfg_dict}

            return optimizer

    return LitLDMUNet(cfg)
