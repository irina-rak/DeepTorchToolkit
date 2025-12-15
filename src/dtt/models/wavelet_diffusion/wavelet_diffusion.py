"""Wavelet Diffusion model for high-resolution medical image synthesis.

This module implements Wavelet Diffusion Models (WDM) which perform diffusion
in wavelet space rather than pixel/voxel space. This approach:
1. Eliminates the need for VAE/VQVAE for latent compression
2. Is memory-efficient for high-resolution 3D images (up to 256^3)
3. Preserves fine details through wavelet decomposition

The model applies DWT to input images, performs diffusion in wavelet space,
and applies IDWT to reconstruct the final images.

References:
    - Friedrich et al., "WDM: 3D Wavelet Diffusion Models for High-Resolution
      Medical Image Synthesis" (DGM4MICCAI 2024)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from dtt.utils.registry import register_model

__all__ = ["build_wavelet_diffusion"]


@register_model("wavelet_diffusion")
def build_wavelet_diffusion(cfg: dict[str, Any]):
    """Build Wavelet Diffusion model from config.

    Expected config structure:
        model:
            name: "wavelet_diffusion"
            optim:
                name: adam | adamw | sgd
                lr: float (e.g., 1e-4)
                weight_decay: float (default: 0.0)
            scheduler:
                name: cosine | reduce_on_plateau | step | exponential | null
                params: {}
            params:
                # UNet architecture
                unet_config:
                    image_size: int (e.g., 128)
                    in_channels: int (1 for raw image, or 4/8 for wavelet channels)
                    model_channels: int (e.g., 64)
                    out_channels: int (same as in_channels)
                    num_res_blocks: int (e.g., 2)
                    attention_resolutions: list[int] (e.g., [16, 8])
                    dropout: float (default: 0.0)
                    channel_mult: list[int] (e.g., [1, 2, 4, 8])
                    num_heads: int (default: 4)
                    num_head_channels: int (default: -1)
                    num_groups: int (default: 32)
                    use_wavelet_updown: bool (default: true)
                    wavelet: str (default: "haar")
                    resblock_updown: bool (default: true)
                    additive_skips: bool (default: false)
                    bottleneck_attention: bool (default: true)

                # Diffusion settings
                spatial_dims: int (2 or 3)
                num_train_timesteps: int (default: 1000)
                beta_start: float (default: 0.0001)
                beta_end: float (default: 0.02)
                beta_schedule: str (default: "linear")

                # Wavelet settings
                wavelet: str (default: "haar")
                apply_wavelet_transform: bool (default: true)
                    If true, applies DWT before diffusion and IDWT after

                # Training settings
                manual_accumulate_grad_batches: int (default: 4)
                seed: int | None (default: None)
                _logging: bool (default: True)
                data_range: str (default: "[-1,1]")

                # EMA settings
                use_ema: bool (default: False)
                ema_decay: float (default: 0.9999)

                # Validation settings
                generate_validation_samples: bool (default: True)
                generate_frequency: int (default: 5)
                val_max_batches: int (default: 3)
                inference_timesteps: int (default: 50)
    """
    # Lazy imports
    import copy
    import os

    import torch
    import torch.nn.functional as F
    from lightning.pytorch import LightningModule
    from monai.metrics import PSNRMetric, SSIMMetric
    from monai.utils import set_determinism
    from torch.nn import MSELoss

    from dtt.models.base import build_optimizer, build_scheduler
    from dtt.models.wavelet_diffusion.dwt_idwt import DWT_2D, DWT_3D, IDWT_2D, IDWT_3D
    from dtt.models.wavelet_diffusion.wavelet_unet import WaveletDiffusionUNet
    from dtt.utils.logging import get_console

    console = get_console()

    class DDPMScheduler:
        """Simple DDPM noise scheduler.

        Implements the forward diffusion process q(x_t | x_0) and
        reverse process p(x_{t-1} | x_t).
        """

        def __init__(
            self,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = "linear",
            device: torch.device = None,
        ):
            self.num_train_timesteps = num_train_timesteps
            self.device = device or torch.device("cpu")

            # Compute beta schedule
            if beta_schedule == "linear":
                self.betas = torch.linspace(
                    beta_start, beta_end, num_train_timesteps, device=self.device
                )
            elif beta_schedule == "cosine":
                steps = num_train_timesteps + 1
                s = 0.008
                x = torch.linspace(0, num_train_timesteps, steps, device=self.device)
                alphas_cumprod = (
                    torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
                )
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                self.betas = torch.clip(betas, 0.0001, 0.9999)
            elif beta_schedule == "scaled_linear":
                self.betas = (
                    torch.linspace(
                        beta_start**0.5, beta_end**0.5, num_train_timesteps, device=self.device
                    )
                    ** 2
                )
            else:
                raise ValueError(f"Unknown beta schedule: {beta_schedule}")

            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

            # Calculations for diffusion q(x_t | x_0) and others
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

            # Calculations for posterior q(x_{t-1} | x_t, x_0)
            self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            self.posterior_log_variance_clipped = torch.log(
                torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
            )
            self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
            )

        def to(self, device: torch.device):
            """Move scheduler tensors to device."""
            self.device = device
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
            self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
            self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
            self.posterior_variance = self.posterior_variance.to(device)
            self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
            self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
            self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
            return self

        def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
        ) -> torch.Tensor:
            """Add noise to samples at given timesteps (forward diffusion)."""
            sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
            sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

            # Expand dimensions for broadcasting
            while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

            noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
            return noisy_samples

        def step(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            """Perform one step of the reverse diffusion process."""
            t = timestep

            # Predict x_0 from noise prediction
            pred_original_sample = (
                self.sqrt_recip_alphas_cumprod[t] * sample
                - self.sqrt_recipm1_alphas_cumprod[t] * model_output
            )

            # NOTE: No clamping here - wavelet coefficients can exceed [-1,1]
            # Clamping is only valid for pixel-space diffusion

            # Compute posterior mean
            posterior_mean = (
                self.posterior_mean_coef1[t] * pred_original_sample
                + self.posterior_mean_coef2[t] * sample
            )

            # Add noise for t > 0
            if t > 0:
                if generator is not None:
                    noise = torch.randn(
                        sample.shape,
                        dtype=sample.dtype,
                        device=sample.device,
                        generator=generator,
                    )
                else:
                    noise = torch.randn_like(sample)
                posterior_variance = self.posterior_variance[t]
                prev_sample = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                prev_sample = posterior_mean

            return prev_sample

    class LitWaveletDiffusion(LightningModule):
        """Lightning module for Wavelet Diffusion."""

        def __init__(self, config: Any):
            super().__init__()

            # Normalize config
            if isinstance(config, dict):
                from dtt.config.schemas import Config as ConfigSchema

                config = ConfigSchema.model_validate(config)

            mcfg = config.model
            p = mcfg.params

            # Disable automatic optimization for manual control
            self.automatic_optimization = False

            # Set seed if provided
            seed = p.get("seed")
            if seed is not None:
                set_determinism(seed=seed)
                console.log(f"[bold][yellow]Setting seed to {seed}.[/yellow][/bold]")

            # Extract configurations
            unet_config = p.get("unet_config", {})
            self.spatial_dims = p.get("spatial_dims", 3)
            self.wavelet = p.get("wavelet", "haar")
            self.apply_wavelet_transform = p.get("apply_wavelet_transform", True)

            # Determine input/output channels based on wavelet transform
            if self.apply_wavelet_transform:
                # DWT produces 4 subbands for 2D, 8 for 3D
                num_subbands = 4 if self.spatial_dims == 2 else 8
                base_channels = unet_config.get("in_channels", 1)
                unet_config["in_channels"] = base_channels * num_subbands
                unet_config["out_channels"] = base_channels * num_subbands
                self.base_channels = base_channels
            else:
                self.base_channels = unet_config.get("in_channels", 1)

            unet_config["dims"] = self.spatial_dims

            # Initialize UNet
            self.model = WaveletDiffusionUNet(**unet_config)

            # Initialize wavelet transforms
            if self.apply_wavelet_transform:
                if self.spatial_dims == 2:
                    self.dwt = DWT_2D(self.wavelet)
                    self.idwt = IDWT_2D(self.wavelet)
                else:
                    self.dwt = DWT_3D(self.wavelet)
                    self.idwt = IDWT_3D(self.wavelet)

            # Initialize noise scheduler
            num_train_timesteps = p.get("num_train_timesteps", 1000)
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=p.get("beta_start", 0.0001),
                beta_end=p.get("beta_end", 0.02),
                beta_schedule=p.get("beta_schedule", "linear"),
            )
            self.num_train_timesteps = num_train_timesteps
            self.inference_timesteps = p.get("inference_timesteps", 50)

            # Loss and metrics
            self.mse_loss = MSELoss()
            self.psnr_metric = PSNRMetric(max_val=1.0)
            self.ssim_metric = SSIMMetric(spatial_dims=self.spatial_dims, data_range=1.0)

            # Training settings
            self.manual_accumulate_grad_batches = p.get("manual_accumulate_grad_batches", 4)
            self._logging = p.get("_logging", True)
            self.data_range = p.get("data_range", "[-1,1]")

            # Validation settings
            self.generate_validation_samples = p.get("generate_validation_samples", True)
            self.generate_frequency = p.get("generate_frequency", 5)
            self.val_max_batches = p.get("val_max_batches", 3)

            # Store optimizer and scheduler configs
            self.optim_cfg = mcfg.optim
            self.scheduler_cfg = mcfg.scheduler

            # Note: EMA is now handled by EMACallback (configured in callbacks.ema)
            # The callback will set self.ema_model at training start if enabled

            self.save_hyperparameters(ignore=["model", "dwt", "idwt"])

            console.log("[bold green]Wavelet Diffusion initialized:[/bold green]")
            console.log(f"  - Spatial dims: {self.spatial_dims}")
            console.log(f"  - Wavelet: {self.wavelet}")
            console.log(f"  - Apply wavelet transform: {self.apply_wavelet_transform}")
            console.log(f"  - Train timesteps: {num_train_timesteps}")

        def _get_inference_model(self):
            """Get the model to use for inference (EMA if available, else main model).

            The EMA model is set by EMACallback during on_fit_start if callbacks.ema
            is configured. The callback uses swa_utils.AveragedModel which properly
            handles both parameters and buffers.
            """
            ema = getattr(self, "ema_model", None)
            if ema is not None:
                return ema
            return self.model

        def _normalize_data(self, x: torch.Tensor) -> torch.Tensor:
            """Normalize data to specified range."""
            if self.data_range == "[-1,1]":
                return x * 2.0 - 1.0
            return x

        def _denormalize_data(self, x: torch.Tensor) -> torch.Tensor:
            """Denormalize data back to [0, 1]."""
            if self.data_range == "[-1,1]":
                return (x + 1.0) / 2.0
            return x

        def _apply_dwt(self, x: torch.Tensor) -> torch.Tensor:
            """Apply discrete wavelet transform and stack subbands."""
            if not self.apply_wavelet_transform:
                return x

            subbands = self.dwt(x)
            # Stack subbands along channel dimension
            return torch.cat(subbands, dim=1)

        def _apply_idwt(self, x: torch.Tensor) -> torch.Tensor:
            """Split stacked subbands and apply inverse wavelet transform."""
            if not self.apply_wavelet_transform:
                return x

            num_subbands = 4 if self.spatial_dims == 2 else 8
            subbands = torch.chunk(x, num_subbands, dim=1)
            return self.idwt(*subbands)

        def on_fit_start(self):
            """Move scheduler to device when training starts."""
            self.scheduler.to(self.device)
            # Note: EMA model device handling is now done by EMACallback

        def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **kwargs) -> torch.Tensor:
            """Forward pass through the model."""
            return self.model(x, timesteps, **kwargs)

        def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            """Training step with noise prediction loss."""
            images = batch["image"]
            images = self._normalize_data(images)

            # Apply wavelet transform
            wavelet_images = self._apply_dwt(images)

            # Get optimizer
            opt = self.optimizers()

            # Determine if we should step the optimizer
            is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
            should_step = (
                (batch_idx + 1) % self.manual_accumulate_grad_batches == 0
            ) or is_last_batch

            # Sample noise and timesteps
            noise = torch.randn_like(wavelet_images)
            timesteps = torch.randint(
                0,
                self.num_train_timesteps,
                (wavelet_images.shape[0],),
                device=self.device,
                dtype=torch.long,
            )

            # Add noise
            noisy_images = self.scheduler.add_noise(wavelet_images, noise, timesteps)

            # Predict noise
            noise_pred = self.forward(noisy_images, timesteps)

            # Compute loss
            loss = self.mse_loss(noise_pred, noise)

            # Scale loss for gradient accumulation
            loss = loss / self.manual_accumulate_grad_batches

            # Backward pass
            self.manual_backward(loss)

            if should_step:
                opt.step()
                opt.zero_grad(set_to_none=True)  # Zero gradients AFTER stepping
                # Note: EMA update is now handled by EMACallback.on_train_batch_end()

            if self._logging:
                self.log(
                    "train/loss",
                    loss * self.manual_accumulate_grad_batches,
                    prog_bar=True,
                    sync_dist=True,
                )

            return {"loss": loss * self.manual_accumulate_grad_batches}

        def validation_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            """Validation step."""
            # Cache first batch for sample generation (avoids DDP deadlock)
            if batch_idx == 0:
                self._cached_val_batch = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            images = batch["image"]
            images = self._normalize_data(images)

            # Apply wavelet transform
            wavelet_images = self._apply_dwt(images)

            # Sample noise and timesteps
            noise = torch.randn_like(wavelet_images)
            timesteps = torch.randint(
                0,
                self.num_train_timesteps,
                (wavelet_images.shape[0],),
                device=self.device,
                dtype=torch.long,
            )

            # Add noise
            noisy_images = self.scheduler.add_noise(wavelet_images, noise, timesteps)

            # Use EMA model for validation if available (set by EMACallback)
            model = self._get_inference_model()
            noise_pred = model(noisy_images, timesteps)

            loss = self.mse_loss(noise_pred, noise)

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
            """Generate samples using diffusion process."""
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            console.log(f"[cyan]Generating {num_samples} samples...[/cyan]")

            # Use EMA model if available (set by EMACallback)
            model_to_use = self._get_inference_model()
            model_to_use.eval()

            # Use cached validation batch (avoids DDP deadlock from dataloader iteration)
            if not hasattr(self, "_cached_val_batch") or self._cached_val_batch is None:
                console.log("[yellow]No cached validation batch, skipping generation[/yellow]")
                return

            images = self._cached_val_batch["image"].to(self.device)[:num_samples]

            # Determine generation shape
            shape = (self.base_channels, *images.shape[2:])

            # Generate samples with mixed precision
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                generated = self.sample(
                    batch_size=num_samples,
                    shape=shape,
                    num_inference_steps=self.inference_timesteps,
                )

            # Get log directory
            log_dir = self._get_log_dir()
            samples_dir = os.path.join(log_dir, "wavelet_diffusion_samples")
            os.makedirs(samples_dir, exist_ok=True)

            # Save comparison images
            wandb_images = []
            for i in range(min(num_samples, images.shape[0])):
                orig = images[i].cpu().squeeze()
                gen = generated[i].cpu().squeeze()

                # Handle 3D: take center slice
                if orig.dim() == 3:
                    center = orig.shape[0] // 2
                    orig = orig[center]
                    gen = gen[center]

                # Normalize for display
                orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
                gen = (gen - gen.min()) / (gen.max() - gen.min() + 1e-8)

                # Rotate for correct orientation
                orig = np.rot90(orig.numpy(), k=-1)
                gen = np.rot90(gen.numpy(), k=-1)

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(orig, cmap="gray")
                axes[0].set_title("Original")
                axes[0].axis("off")

                axes[1].imshow(gen, cmap="gray")
                axes[1].set_title("Generated")
                axes[1].axis("off")

                plt.suptitle(f"Epoch {self.current_epoch}")
                plt.tight_layout()
                save_path = os.path.join(
                    samples_dir, f"epoch_{self.current_epoch:04d}_sample_{i}.png"
                )
                plt.savefig(save_path, bbox_inches="tight", dpi=100)
                plt.close(fig)

                # Collect for W&B logging
                comparison = np.concatenate([orig, gen], axis=1)
                wandb_images.append(comparison)

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

            console.log(f"[dim]Saved wavelet diffusion samples to: {samples_dir}[/dim]")

        def _get_log_dir(self) -> str:
            """Get the logging directory."""
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

        @torch.no_grad()
        def sample(
            self,
            batch_size: int,
            shape: tuple[int, ...],
            num_inference_steps: int | None = None,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            """Generate samples using the reverse diffusion process.

            Args:
                batch_size: Number of samples to generate
                shape: Shape of the output (C, *spatial_dims) in wavelet space
                num_inference_steps: Number of denoising steps (default: inference_timesteps)
                generator: Random generator for reproducibility

            Returns:
                Generated samples in original image space
            """
            num_inference_steps = num_inference_steps or self.inference_timesteps
            model = self._get_inference_model()
            model.eval()

            # Determine wavelet space shape
            if self.apply_wavelet_transform:
                num_subbands = 4 if self.spatial_dims == 2 else 8
                wavelet_shape = (self.base_channels * num_subbands, *[s // 2 for s in shape[1:]])
            else:
                wavelet_shape = shape

            # Start from pure noise
            sample = torch.randn(
                batch_size, *wavelet_shape, device=self.device, generator=generator
            )

            # Define timestep schedule for inference
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = list(range(0, self.num_train_timesteps, step_ratio))[::-1]

            for t in timesteps:
                # Predict noise
                t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                noise_pred = model(sample, t_tensor)

                # Denoise step
                sample = self.scheduler.step(noise_pred, t, sample, generator=generator)

            # Apply inverse wavelet transform
            sample = self._apply_idwt(sample)

            # Denormalize
            sample = self._denormalize_data(sample)
            sample = torch.clamp(sample, 0.0, 1.0)

            return sample

        def test_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            """Test step: Generate samples and compute metrics."""
            batch_size = batch["image"].shape[0]
            device = self.device

            # Get original images
            imgs_original = batch["image"].to(device)

            # Determine shape for generation
            if self.spatial_dims == 2:
                shape = (self.base_channels, *imgs_original.shape[2:])
            else:
                shape = (self.base_channels, *imgs_original.shape[2:])

            # Generate samples
            generated_samples = self.sample(
                batch_size=batch_size,
                shape=shape,
                num_inference_steps=self.inference_timesteps,
            )

            # Compute metrics
            psnr = self.psnr_metric(generated_samples, imgs_original).mean()
            ssim = self.ssim_metric(generated_samples, imgs_original).mean()

            metrics = {
                "psnr": psnr.item() if isinstance(psnr, torch.Tensor) else psnr,
                "ssim": ssim.item() if isinstance(ssim, torch.Tensor) else ssim,
            }

            if self._logging:
                self.log("test/psnr", metrics["psnr"], sync_dist=True)
                self.log("test/ssim", metrics["ssim"], sync_dist=True)

            # Save generated samples if output directory is set
            if hasattr(self, "inference_output_dir"):
                self._save_samples(generated_samples, batch_idx)

            return metrics

        def _save_samples(self, samples: torch.Tensor, batch_idx: int):
            """Save generated samples to disk."""
            output_dir = getattr(self, "inference_output_dir", None)
            if output_dir is None:
                return

            samples_dir = os.path.join(output_dir, "generated_samples")
            os.makedirs(samples_dir, exist_ok=True)

            for i in range(samples.shape[0]):
                sample = samples[i].cpu()
                sample_path = os.path.join(samples_dir, f"sample_batch{batch_idx}_idx{i}.pt")
                torch.save(sample, sample_path)

        def configure_optimizers(self):
            """Configure optimizer and scheduler."""
            optimizer = build_optimizer(self.parameters(), self.optim_cfg)
            scheduler = build_scheduler(optimizer, self.scheduler_cfg)

            if scheduler is not None:
                scheduler_cfg = {"scheduler": scheduler}
                if self.scheduler_cfg.name and "plateau" in self.scheduler_cfg.name.lower():
                    scheduler_cfg["monitor"] = "val/loss"
                    scheduler_cfg["interval"] = "epoch"
                return {"optimizer": optimizer, "lr_scheduler": scheduler_cfg}

            return optimizer

    return LitWaveletDiffusion(cfg)
