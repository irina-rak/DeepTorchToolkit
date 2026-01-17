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
    import os

    import torch
    import torch.nn as nn
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
            prediction_type: str = "epsilon",
            device: torch.device = None,
        ):
            """Initialize DDPM scheduler.

            Args:
                prediction_type: 'epsilon' (predict noise) or 'x_start' (predict x0)
            """
            self.num_train_timesteps = num_train_timesteps
            self.prediction_type = prediction_type
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

            # Get x_0 prediction based on prediction_type
            if self.prediction_type == "x_start":
                # Model directly predicts x_0
                pred_original_sample = model_output
            else:
                # Model predicts noise (epsilon), convert to x_0
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

    class SubbandNormalizer(nn.Module):
        """Normalizes wavelet subbands to have balanced energy.

        The problem: In wavelet decomposition of natural images, the LLL
        (low-frequency) subband contains ~95-99% of the total energy, while
        the 7 detail subbands contain only ~1-5%. When uniform Gaussian noise
        is added during diffusion, the detail subbands become noise-dominated
        (SNR << 0 dB), making it impossible for the model to learn them.

        The solution: Normalize each subband independently to have unit variance
        before the diffusion process, then denormalize after reconstruction.
        This ensures all subbands have equal signal-to-noise ratio during training.

        Args:
            num_subbands: Number of wavelet subbands (4 for 2D, 8 for 3D)
            momentum: Momentum for running statistics update (like BatchNorm)
            eps: Small constant for numerical stability
        """

        def __init__(
            self,
            num_subbands: int = 8,
            momentum: float = 0.1,
            eps: float = 1e-6,
        ):
            super().__init__()
            self.num_subbands = num_subbands
            self.momentum = momentum
            self.eps = eps

            # Running statistics (like BatchNorm)
            # These track the mean and std of each subband across training
            self.register_buffer("running_mean", torch.zeros(num_subbands))
            self.register_buffer("running_std", torch.ones(num_subbands))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

            # Flag to control whether to update running stats
            self.track_running_stats = True

        def normalize(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
            """Normalize stacked subbands to have zero mean and unit variance.

            Args:
                x: Stacked subbands tensor of shape (B, num_subbands*C, *spatial)
                   where C is the number of input channels (typically 1)
                update_stats: Whether to update running statistics

            Returns:
                Normalized tensor with same shape
            """
            batch_size = x.shape[0]
            channels_per_subband = x.shape[1] // self.num_subbands
            spatial_dims = x.shape[2:]

            # Reshape to (B, num_subbands, C, *spatial) for per-subband stats
            x_reshaped = x.view(batch_size, self.num_subbands, channels_per_subband, *spatial_dims)

            if self.training and update_stats and self.track_running_stats:
                # Compute batch statistics (mean and std per subband)
                # Reduce over batch, channel, and spatial dimensions
                with torch.no_grad():
                    batch_mean = x_reshaped.mean(dim=(0, 2, *range(3, 3 + len(spatial_dims))))
                    batch_var = x_reshaped.var(
                        dim=(0, 2, *range(3, 3 + len(spatial_dims))), unbiased=False
                    )
                    batch_std = torch.sqrt(batch_var + self.eps)

                    # Update running stats with exponential moving average
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * batch_mean
                    self.running_std = (
                        1 - self.momentum
                    ) * self.running_std + self.momentum * batch_std
                    self.num_batches_tracked += 1

            # Use running stats for normalization (both training and inference)
            # This ensures consistent normalization between train/val/test
            mean = self.running_mean.view(1, self.num_subbands, 1, *([1] * len(spatial_dims)))
            std = self.running_std.view(1, self.num_subbands, 1, *([1] * len(spatial_dims)))

            x_normalized = (x_reshaped - mean) / (std + self.eps)

            # Reshape back to (B, num_subbands*C, *spatial)
            return x_normalized.view(batch_size, -1, *spatial_dims)

        def denormalize(self, x: torch.Tensor) -> torch.Tensor:
            """Denormalize subbands back to original scale.

            Args:
                x: Normalized stacked subbands tensor

            Returns:
                Denormalized tensor with original scale
            """
            batch_size = x.shape[0]
            channels_per_subband = x.shape[1] // self.num_subbands
            spatial_dims = x.shape[2:]

            # Reshape to (B, num_subbands, C, *spatial)
            x_reshaped = x.view(batch_size, self.num_subbands, channels_per_subband, *spatial_dims)

            # Apply inverse normalization using running stats
            mean = self.running_mean.view(1, self.num_subbands, 1, *([1] * len(spatial_dims)))
            std = self.running_std.view(1, self.num_subbands, 1, *([1] * len(spatial_dims)))

            x_denormalized = x_reshaped * (std + self.eps) + mean

            # Reshape back to (B, num_subbands*C, *spatial)
            return x_denormalized.view(batch_size, -1, *spatial_dims)

        def get_stats(self) -> dict:
            """Get current running statistics for debugging."""
            return {
                "running_mean": self.running_mean.cpu().numpy(),
                "running_std": self.running_std.cpu().numpy(),
                "num_batches_tracked": self.num_batches_tracked.item(),
            }

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
                in_channels = unet_config.get("in_channels", 1)
                
                # Check if channels are already wavelet-transformed (from checkpoint reload)
                # If in_channels is already a multiple of num_subbands and > 1, it's likely
                # already been transformed. We store base_channels separately to detect this.
                if in_channels == num_subbands or in_channels == num_subbands * 2:
                    # Already transformed (e.g., 4 for 2D with 1 base channel, 8 for 3D)
                    self.base_channels = in_channels // num_subbands
                    console.log(f"[dim]Detected pre-transformed channels: {in_channels} -> base_channels={self.base_channels}[/dim]")
                else:
                    # Not yet transformed, apply multiplication
                    self.base_channels = in_channels
                    unet_config["in_channels"] = in_channels * num_subbands
                    unet_config["out_channels"] = in_channels * num_subbands
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

                # Subband normalization to balance energy across subbands
                # This is critical for proper diffusion training:
                # - LLL subband typically contains 95-99% of energy
                # - Detail subbands become noise-dominated without normalization
                self.normalize_subbands = p.get("normalize_subbands", True)
                if self.normalize_subbands:
                    num_subbands = 4 if self.spatial_dims == 2 else 8
                    self.subband_normalizer = SubbandNormalizer(
                        num_subbands=num_subbands,
                        momentum=p.get("subband_norm_momentum", 0.1),
                        eps=1e-6,
                    )
                else:
                    self.subband_normalizer = None

                # Per-subband loss weighting to emphasize high-frequency details
                # Default weights: higher weight for high-frequency subbands
                # Order for 3D: [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]
                # Order for 2D: [LL, LH, HL, HH]
                default_weights_3d = [
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    3.0,
                    4.0,
                ]  # Emphasize detail subbands
                default_weights_2d = [1.0, 2.0, 2.0, 4.0]
                default_weights = (
                    default_weights_2d if self.spatial_dims == 2 else default_weights_3d
                )

                subband_loss_weights = p.get("subband_loss_weights", default_weights)
                if subband_loss_weights is not None:
                    self.subband_loss_weights = torch.tensor(
                        subband_loss_weights, dtype=torch.float32
                    )
                    # Normalize weights so they sum to num_subbands (preserves average loss magnitude)
                    self.subband_loss_weights = self.subband_loss_weights * (
                        num_subbands / self.subband_loss_weights.sum()
                    )
                    console.log(f"  - Subband loss weights: {self.subband_loss_weights.tolist()}")
                else:
                    self.subband_loss_weights = None
            else:
                self.normalize_subbands = False
                self.subband_normalizer = None
                self.subband_loss_weights = None

            # Initialize noise scheduler
            num_train_timesteps = p.get("num_train_timesteps", 1000)
            self.prediction_type = p.get("prediction_type", "epsilon")  # 'epsilon' or 'x_start'
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=p.get("beta_start", 0.0001),
                beta_end=p.get("beta_end", 0.02),
                beta_schedule=p.get("beta_schedule", "linear"),
                prediction_type=self.prediction_type,
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

            # Mask conditioning settings (for conditional generation)
            # cfg_dropout_prob: probability of dropping mask during training (classifier-free guidance)
            # guidance_scale: strength of conditioning at inference (1.0 = no guidance, higher = stronger)
            self.mask_conditioning = unet_config.get("context_dim") is not None
            self.cfg_dropout_prob = p.get("cfg_dropout_prob", 0.1)  # 10% dropout for CFG
            self.guidance_scale = p.get("guidance_scale", 1.0)  # Configurable at inference
            self.enable_cfg = p.get("enable_cfg", True)  # Option to disable CFG entirely

            # Store optimizer and scheduler configs
            self.optim_cfg = mcfg.optim
            self.scheduler_cfg = mcfg.scheduler

            # Note: EMA is now handled by EMACallback (configured in callbacks.ema)
            # The callback will set self.ema_model at training start if enabled

            self.save_hyperparameters(ignore=["model", "dwt", "idwt", "subband_normalizer"])

            console.log("[bold green]Wavelet Diffusion initialized:[/bold green]")
            console.log(f"  - Spatial dims: {self.spatial_dims}")
            console.log(f"  - Wavelet: {self.wavelet}")
            console.log(f"  - Apply wavelet transform: {self.apply_wavelet_transform}")
            console.log(f"  - Normalize subbands: {self.normalize_subbands}")
            console.log(f"  - Train timesteps: {num_train_timesteps}")
            if self.mask_conditioning:
                console.log(f"  - Mask conditioning: enabled (CFG dropout={self.cfg_dropout_prob})")

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

        def _apply_dwt(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
            """Apply discrete wavelet transform and stack subbands.

            Args:
                x: Input tensor of shape (B, C, *spatial)
                update_stats: Whether to update running statistics in normalizer

            Returns:
                Stacked (and optionally normalized) subbands
            """
            if not self.apply_wavelet_transform:
                return x

            # Apply DWT and get subbands
            subbands = self.dwt(x)

            # Scale the low-frequency subband (LL/LLL) by 1/3 to balance training
            # This allows high-frequency subbands to be trained correctly
            # (otherwise their loss will struggle to converge due to energy imbalance)
            if self.spatial_dims == 2:
                # 2D: 4 subbands (LL, LH, HL, HH)
                LL, LH, HL, HH = subbands  # noqa F806
                stacked = torch.cat([LL, LH, HL, HH], dim=1)
            else:
                # 3D: 8 subbands (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = subbands  # noqa F806
                stacked = torch.cat(
                    [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1
                )  # noqa F806

            # Apply per-subband normalization if enabled
            if self.normalize_subbands and self.subband_normalizer is not None:
                stacked = self.subband_normalizer.normalize(stacked, update_stats=update_stats)

            return stacked

        def _apply_idwt(self, x: torch.Tensor) -> torch.Tensor:
            """Split stacked subbands and apply inverse wavelet transform.

            Args:
                x: Stacked (and optionally normalized) subbands

            Returns:
                Reconstructed image in spatial domain
            """
            if not self.apply_wavelet_transform:
                return x

            # Denormalize subbands before IDWT if normalization was applied
            if self.normalize_subbands and self.subband_normalizer is not None:
                x = self.subband_normalizer.denormalize(x)

            num_subbands = 4 if self.spatial_dims == 2 else 8
            subbands = torch.chunk(x, num_subbands, dim=1)
            subbands = (subbands[0], *subbands[1:])
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

            # Handle mask conditioning with classifier-free guidance dropout
            mask = batch.get("label", None)
            if mask is not None and self.mask_conditioning:
                # Apply CFG dropout: randomly drop mask to enable unconditional generation
                if self.enable_cfg and self.training:
                    drop_mask = torch.rand(mask.shape[0], device=mask.device) < self.cfg_dropout_prob
                    # Set mask to None for dropped samples by zeroing it out
                    mask = mask * (~drop_mask).float().view(-1, 1, *([1] * (mask.dim() - 2)))
            elif not self.mask_conditioning:
                mask = None  # Ignore mask if conditioning not enabled

            # Predict (noise or x0 depending on prediction_type)
            model_output = self.forward(noisy_images, timesteps, mask=mask)

            # Determine target based on prediction_type
            if self.prediction_type == "x_start":
                target = wavelet_images  # Model predicts clean x0
            else:
                target = noise  # Model predicts noise (epsilon)

            # Compute loss - use per-subband weighting if enabled
            if self.subband_loss_weights is not None and self.apply_wavelet_transform:
                # Compute weighted per-subband MSE loss
                num_subbands = 4 if self.spatial_dims == 2 else 8
                weights = self.subband_loss_weights.to(self.device)

                # Compute MSE per subband and apply weights
                # Shape: [B, num_subbands, ...] -> compute MSE per subband
                loss = 0.0
                for i in range(num_subbands):
                    subband_pred = model_output[:, i : i + 1]
                    subband_target = target[:, i : i + 1]
                    subband_mse = F.mse_loss(subband_pred, subband_target)
                    loss = loss + weights[i] * subband_mse

                # Average across subbands (weights already normalized to sum to num_subbands)
                loss = loss / num_subbands
            else:
                loss = self.mse_loss(model_output, target)

            # Scale loss for gradient accumulation
            loss = loss / self.manual_accumulate_grad_batches

            # Backward pass
            self.manual_backward(loss)

            if should_step:
                opt.step()
                opt.zero_grad(set_to_none=True)  # Zero gradients AFTER stepping
                # Note: EMA update is now handled by EMACallback.on_train_batch_end()

                # Manual scheduler step (required with automatic_optimization=False)
                # Step the scheduler at the end of each epoch (when last batch is processed)
                if is_last_batch:
                    sch = self.lr_schedulers()
                    if sch is not None:
                        sch.step()

            if self._logging:
                self.log(
                    "train/loss",
                    loss * self.manual_accumulate_grad_batches,
                    prog_bar=True,
                    sync_dist=True,
                )

                # Log per-subband loss periodically (every 50 batches) to monitor balanced learning
                if batch_idx % 50 == 0 and self.apply_wavelet_transform:
                    num_subbands = 4 if self.spatial_dims == 2 else 8
                    subband_names = (
                        ["LL", "LH", "HL", "HH"]
                        if self.spatial_dims == 2
                        else ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                    )
                    with torch.no_grad():
                        # Compute per-subband MSE
                        for i, name in enumerate(subband_names):
                            subband_pred = model_output[:, i : i + 1]
                            subband_target = target[:, i : i + 1]
                            subband_mse = F.mse_loss(subband_pred, subband_target)
                            self.log(
                                f"train/subband_loss/{name}",
                                subband_mse,
                                sync_dist=True,
                            )

                # Log subband normalizer statistics periodically (every 100 batches)
                if (
                    self.normalize_subbands
                    and self.subband_normalizer is not None
                    and batch_idx % 100 == 0
                ):
                    stats = self.subband_normalizer.get_stats()
                    subband_names = (
                        ["LL", "LH", "HL", "HH"]
                        if self.spatial_dims == 2
                        else ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                    )
                    for i, name in enumerate(subband_names):
                        self.log(
                            f"subband_std/{name}",
                            stats["running_std"][i],
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
                    k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }

            images = batch["image"]
            images = self._normalize_data(images)

            # Apply wavelet transform (don't update normalizer stats during validation)
            wavelet_images = self._apply_dwt(images, update_stats=False)

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

            # Get mask for conditioning (no CFG dropout during validation)
            mask = batch.get("label", None)
            if not self.mask_conditioning:
                mask = None

            # Use EMA model for validation if available (set by EMACallback)
            model = self._get_inference_model()
            model_output = model(noisy_images, timesteps, mask=mask)

            # Determine target based on prediction_type
            if self.prediction_type == "x_start":
                target = wavelet_images  # Model predicts clean x0
            else:
                target = noise  # Model predicts noise (epsilon)

            loss = self.mse_loss(model_output, target)

            if self._logging:
                self.log("val/loss", loss, prog_bar=True, sync_dist=True)

                # Log per-subband validation loss
                if self.apply_wavelet_transform:
                    subband_names = (
                        ["LL", "LH", "HL", "HH"]
                        if self.spatial_dims == 2
                        else ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                    )
                    with torch.no_grad():
                        for i, name in enumerate(subband_names):
                            subband_pred = model_output[:, i : i + 1]
                            subband_target = target[:, i : i + 1]
                            subband_mse = F.mse_loss(subband_pred, subband_target)
                            self.log(
                                f"val/subband_loss/{name}",
                                subband_mse,
                                sync_dist=True,
                            )

            return {"loss": loss}

        def on_validation_epoch_end(self) -> None:
            """Generate samples at the end of validation epoch."""
            if not self.trainer.is_global_zero:
                return

            should_generate = (
                self.generate_validation_samples
                and self.current_epoch > 0  # Skip epoch 0
                and (self.current_epoch == 1 or self.current_epoch % self.generate_frequency == 0)
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

            # Compute image-space quality metrics on generated samples
            # Note: For unconditional generation, these measure distributional quality,
            # not exact match (since generated != original by design)
            try:
                # Compute metrics on full 3D volumes (not slices)
                with torch.no_grad():
                    # Ensure both are in [0, 1] range for metrics
                    gen_norm = generated.clamp(0, 1)
                    orig_norm = images.clamp(0, 1)

                    # PSNR - measures overall intensity difference
                    psnr_val = self.psnr_metric(gen_norm, orig_norm).mean()

                    # SSIM - measures structural similarity
                    ssim_val = self.ssim_metric(gen_norm, orig_norm).mean()

                    # Log without sync_dist since only rank 0 runs _generate_samples
                    # Using sync_dist=True would cause DDP deadlock
                    if self._logging:
                        self.log("val/gen_psnr", psnr_val, sync_dist=False, rank_zero_only=True)
                        self.log("val/gen_ssim", ssim_val, sync_dist=False, rank_zero_only=True)

                    console.log(
                        f"[cyan]Generation metrics: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}[/cyan]"
                    )
            except Exception as e:
                console.log(f"[yellow]Could not compute generation metrics: {e}[/yellow]")

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
            mask: torch.Tensor | None = None,
            guidance_scale: float | None = None,
        ) -> torch.Tensor:
            """Generate samples using the reverse diffusion process.

            Args:
                batch_size: Number of samples to generate
                shape: Shape of the output (C, *spatial_dims) in wavelet space
                num_inference_steps: Number of denoising steps (default: inference_timesteps)
                generator: Random generator for reproducibility
                mask: Optional conditioning mask of shape (B, mask_channels, *spatial_dims)
                guidance_scale: CFG guidance scale (default: self.guidance_scale).
                               1.0 = no guidance, higher = stronger conditioning

            Returns:
                Generated samples in original image space
            """
            guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale

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

                # Apply classifier-free guidance if mask provided and scale > 1
                if mask is not None and self.mask_conditioning and guidance_scale > 1.0:
                    # Conditional prediction (with mask)
                    noise_pred_cond = model(sample, t_tensor, mask=mask)
                    # Unconditional prediction (without mask)
                    noise_pred_uncond = model(sample, t_tensor, mask=None)
                    # CFG blending: uncond + scale * (cond - uncond)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # Standard prediction (with or without mask)
                    noise_pred = model(sample, t_tensor, mask=mask)

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
            """Test step: Generate samples and optionally compute metrics.

            Supports two modes:
            - Conditional (default): Uses test data, computes PSNR/SSIM, saves comparisons
            - Unconditional: Generates from noise only, no metrics, no comparisons

            Mode is determined by self.inference_mode set by the inference runner.
            """
            device = self.device
            is_unconditional = getattr(self, "inference_mode", "conditional") == "unconditional"
            save_comparison = (
                getattr(self, "inference_save_comparison", True) and not is_unconditional
            )

            # Get batch data
            imgs_original = batch["image"].to(device)

            # Determine shape for generation from batch (use spatial dims only)
            spatial_shape = imgs_original.shape[2:]  # Skip batch and channel dims
            shape = (self.base_channels, *spatial_shape)

            # Get mask for conditional generation (if available)
            mask = batch.get("label", None)
            if mask is not None:
                mask = mask.to(device)
            if not self.mask_conditioning:
                mask = None

            # Generate samples (with mask conditioning if enabled)
            generated_samples = self.sample(
                batch_size=imgs_original.shape[0],
                shape=shape,
                num_inference_steps=self.inference_timesteps,
                mask=mask,
                guidance_scale=self.guidance_scale,
            )

            # Compute metrics only for conditional mode (when we have real images to compare)
            metrics = {}
            if not is_unconditional:
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
                self._save_samples(
                    generated_samples,
                    batch_idx,
                    originals=imgs_original if save_comparison else None,
                    save_comparison=save_comparison,
                )

            return metrics

        def _save_samples(
            self,
            samples: torch.Tensor,
            batch_idx: int,
            originals: torch.Tensor | None = None,
            save_comparison: bool = True,
        ):
            """Save generated samples to disk.

            Args:
                samples: Generated samples tensor [B, C, ...]
                batch_idx: Current batch index
                originals: Original images for comparison (optional)
                save_comparison: Whether to save side-by-side comparison images
            """
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            output_dir = getattr(self, "inference_output_dir", None)
            if output_dir is None:
                return

            samples_dir = os.path.join(output_dir, "generated_samples")
            os.makedirs(samples_dir, exist_ok=True)

            for i in range(samples.shape[0]):
                sample = samples[i].cpu()
                sample_idx = batch_idx * samples.shape[0] + i

                # Save as PNG image at native resolution
                img = sample.squeeze()
                if img.dim() == 3:  # 3D volume - take center slice
                    center = img.shape[0] // 2
                    img = img[center]

                # Normalize to [0, 255] for saving
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = np.rot90(img.float().numpy(), k=-1)  # .float() for bf16 compatibility
                img_uint8 = (img * 255).astype(np.uint8)

                # Save using PIL at native resolution
                from PIL import Image

                png_path = os.path.join(samples_dir, f"sample_{sample_idx:05d}.png")
                Image.fromarray(img_uint8, mode="L").save(png_path)

                # Save comparison if requested and originals provided
                if save_comparison and originals is not None:
                    orig = originals[i].cpu().squeeze()
                    if orig.dim() == 3:  # 3D
                        center = orig.shape[0] // 2
                        orig = orig[center]

                    orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
                    orig = np.rot90(orig.numpy(), k=-1)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(orig, cmap="gray")
                    axes[0].set_title("Original")
                    axes[0].axis("off")

                    axes[1].imshow(img, cmap="gray")
                    axes[1].set_title("Generated")
                    axes[1].axis("off")

                    plt.tight_layout()
                    comparison_path = os.path.join(samples_dir, f"comparison_{sample_idx:05d}.png")
                    plt.savefig(comparison_path, bbox_inches="tight", dpi=150)
                    plt.close()

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
