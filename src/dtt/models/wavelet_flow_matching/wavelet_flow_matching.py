"""Wavelet Flow Matching model for high-resolution medical image synthesis.

This module implements Wavelet Flow Matching (WFM) which combines:
1. Wavelet transform compression (from Wavelet Diffusion Models)
2. Flow matching dynamics (from Optimal Transport Flow Matching)

This approach:
- Eliminates the need for VAE/VQVAE for latent compression
- Is memory-efficient for high-resolution 3D images
- Preserves fine details through wavelet decomposition
- Provides faster sampling than diffusion models (10-20 steps vs 1000)

The model applies DWT to input images, performs flow matching in wavelet space,
and applies IDWT to reconstruct the final images.

References:
    - Friedrich et al., "WDM: 3D Wavelet Diffusion Models for High-Resolution
      Medical Image Synthesis" (DGM4MICCAI 2024)
    - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from dtt.utils.registry import register_model

__all__ = ["build_wavelet_flow_matching"]


@register_model("wavelet_flow_matching")
def build_wavelet_flow_matching(cfg: dict[str, Any]):
    """Build Wavelet Flow Matching model from config.

    Expected config structure:
        model:
            name: "wavelet_flow_matching"
            optim:
                name: adam | adamw | sgd
                lr: float (e.g., 3e-4)
                weight_decay: float (default: 0.0)
            scheduler:
                name: cosine | reduce_on_plateau | step | exponential | null
                params: {}
            params:
                # UNet architecture
                unet_config:
                    image_size: int (e.g., 128)
                    in_channels: int (1 for raw image, or 4/8 for wavelet channels)
                    model_channels: int (e.g., 128)
                    out_channels: int (same as in_channels)
                    num_res_blocks: int (e.g., 3)
                    attention_resolutions: list[int] (e.g., [16, 8])
                    dropout: float (default: 0.0)
                    channel_mult: list[int] (e.g., [1, 2, 4])
                    num_heads: int (default: 4)
                    num_head_channels: int (default: -1)
                    num_groups: int (default: 32)
                    use_wavelet_updown: bool (default: false)
                    wavelet: str (default: "haar")
                    resblock_updown: bool (default: true)
                    additive_skips: bool (default: false)
                    bottleneck_attention: bool (default: true)

                # Wavelet settings
                spatial_dims: int (2 or 3)
                wavelet: str (default: "haar")
                apply_wavelet_transform: bool (default: true)

                # Flow matching settings
                solver_args:
                    time_points: int (default: 20) - ODE steps for training
                    method: str (default: "midpoint") - ODE solver method
                max_timestep: int (default: 1000)
                    Maximum timestep value for rescaling t from [0,1] to [0, max_timestep]

                # Training settings
                manual_accumulate_grad_batches: int (default: 4)
                seed: int | None (default: None)
                _logging: bool (default: True)
                data_range: str (default: "[-1,1]")

                # Validation generation settings
                generate_validation_samples: bool (default: True)
                generate_frequency: int (default: 10)
                val_max_batches: int (default: 2)
                val_time_points: int (default: 10)
    """
    # Lazy imports
    import os

    import torch
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import CondOTScheduler
    from flow_matching.solver import ODESolver
    from flow_matching.utils.model_wrapper import ModelWrapper
    from lightning.pytorch import LightningModule
    from monai.metrics import PSNRMetric, SSIMMetric
    from monai.utils import set_determinism
    from torch.nn import MSELoss

    from dtt.models.base import build_optimizer, build_scheduler
    from dtt.models.wavelet_diffusion.dwt_idwt import DWT_2D, DWT_3D, IDWT_2D, IDWT_3D
    from dtt.models.wavelet_diffusion.wavelet_unet import WaveletDiffusionUNet
    from dtt.utils.logging import get_console

    import torch.nn.functional as F

    console = get_console()

    class WaveletUNetWrapper(ModelWrapper):
        """Wrapper for WaveletDiffusionUNet to work with flow matching library.

        This adapter:
        - Converts flow_matching's interface: forward(x, t, **extras)
        - To WaveletDiffusionUNet's interface: forward(x, timesteps, context)
        - Rescales timesteps from [0, 1] to [0, max_timestep]
        
        IMPORTANT: This wrapper does NOT apply DWT/IDWT!
        Flow matching operates entirely in wavelet space.
        DWT should be applied to data before training/inference,
        and IDWT should be applied after generation.
        """

        def __init__(
            self,
            unet: WaveletDiffusionUNet,
            max_timestep: int = 1000,
            apply_wavelet_transform: bool = True,
            dwt=None,
            idwt=None,
            spatial_dims: int = 2,
        ):
            """Initialize the wrapper.

            Args:
                unet: WaveletDiffusionUNet instance
                max_timestep: Maximum timestep value for rescaling
                apply_wavelet_transform: Whether to apply DWT/IDWT
                dwt: DWT transform (required if apply_wavelet_transform=True)
                idwt: IDWT transform (required if apply_wavelet_transform=True)
                spatial_dims: Spatial dimensions (2 or 3)
            """
            super().__init__(model=unet)
            self.max_timestep = max_timestep
            self.apply_wavelet_transform = apply_wavelet_transform
            self.dwt = dwt
            self.idwt = idwt
            self.spatial_dims = spatial_dims

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

        def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            **extras,
        ) -> torch.Tensor:
            """Forward pass with interface adaptation and timestep rescaling.

            Args:
                x: Input tensor IN WAVELET SPACE (batch_size, channels, *spatial_dims)
                t: Timesteps in [0, 1] (batch_size,)
                **extras: Additional arguments (context, etc.)

            Returns:
                Model output IN WAVELET SPACE (same shape as x)
            
            IMPORTANT: DWT/IDWT should NOT be applied here!
            The flow matching operates entirely in wavelet space.
            DWT is applied before calling the solver, IDWT after generation.
            """
            # Rescale t from [0, 1] to [0, max_timestep - 1] and convert to discrete timesteps
            # This matches MOTFM reference implementation and works properly with sinusoidal embeddings
            t_scaled = t * (self.max_timestep - 1)
            t_scaled = t_scaled.floor().long()

            # Ensure timesteps is 1D
            if t_scaled.dim() == 0:
                batch_size = x.shape[0]
                t_scaled = t_scaled.expand(batch_size)
            elif t_scaled.dim() > 1:
                t_scaled = t_scaled.flatten()

            # Extract conditioning if provided
            context = extras.get("context") or extras.get("cond")

            # Call WaveletDiffusionUNet (operates in wavelet space)
            output = self.model(x=x, timesteps=t_scaled, y=context)

            return output

    class SubbandNormalizer(torch.nn.Module):
        """Normalizes wavelet subbands to have balanced energy.

        The problem: In wavelet decomposition of natural images, the LLL
        (low-frequency) subband contains ~95-99% of the total energy, while
        the 7 detail subbands contain only ~1-5%. When uniform Gaussian noise
        is used as the source distribution, the detail subbands
        become noise-dominated (SNR << 0 dB), making it impossible for the model
        to learn them.

        The solution: Normalize each subband independently to have unit variance
        before flow matching, then denormalize after generation.
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
            B = x.shape[0]
            channels_per_subband = x.shape[1] // self.num_subbands
            spatial_dims = x.shape[2:]

            # Reshape to (B, num_subbands, C, *spatial) for per-subband stats
            x_reshaped = x.view(B, self.num_subbands, channels_per_subband, *spatial_dims)

            if self.training and update_stats and self.track_running_stats:
                # Compute batch statistics (mean and std per subband)
                # Reduce over batch, channel, and spatial dimensions
                with torch.no_grad():
                    batch_mean = x_reshaped.mean(dim=(0, 2, *range(3, 3 + len(spatial_dims))))
                    batch_var = x_reshaped.var(dim=(0, 2, *range(3, 3 + len(spatial_dims))), unbiased=False)
                    batch_std = torch.sqrt(batch_var + self.eps)

                    # Update running stats with exponential moving average
                    self.running_mean = (
                        (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                    )
                    self.running_std = (
                        (1 - self.momentum) * self.running_std + self.momentum * batch_std
                    )
                    self.num_batches_tracked += 1

            # Use running stats for normalization (both training and inference)
            # This ensures consistent normalization between train/val/test
            mean = self.running_mean.view(1, self.num_subbands, 1, *([1] * len(spatial_dims)))
            std = self.running_std.view(1, self.num_subbands, 1, *([1] * len(spatial_dims)))

            x_normalized = (x_reshaped - mean) / (std + self.eps)

            # Reshape back to (B, num_subbands*C, *spatial)
            return x_normalized.view(B, -1, *spatial_dims)

        def denormalize(self, x: torch.Tensor) -> torch.Tensor:
            """Denormalize subbands back to original scale.

            Args:
                x: Normalized stacked subbands tensor

            Returns:
                Denormalized tensor with original scale
            """
            B = x.shape[0]
            channels_per_subband = x.shape[1] // self.num_subbands
            spatial_dims = x.shape[2:]

            # Reshape to (B, num_subbands, C, *spatial)
            x_reshaped = x.view(B, self.num_subbands, channels_per_subband, *spatial_dims)

            # Apply inverse normalization using running stats
            mean = self.running_mean.view(1, self.num_subbands, 1, *([1] * len(spatial_dims)))
            std = self.running_std.view(1, self.num_subbands, 1, *([1] * len(spatial_dims)))

            x_denormalized = x_reshaped * (std + self.eps) + mean

            # Reshape back to (B, num_subbands*C, *spatial)
            return x_denormalized.view(B, -1, *spatial_dims)

        def get_stats(self) -> dict:
            """Get current running statistics for debugging."""
            return {
                "running_mean": self.running_mean.cpu().numpy(),
                "running_std": self.running_std.cpu().numpy(),
                "num_batches_tracked": self.num_batches_tracked.item(),
            }

    class LitWaveletFlowMatching(LightningModule):
        """Lightning module for Wavelet Flow Matching."""

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
            self.spatial_dims = p.get("spatial_dims", 2)
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
            unet = WaveletDiffusionUNet(**unet_config)

            # Initialize wavelet transforms
            if self.apply_wavelet_transform:
                if self.spatial_dims == 2:
                    self.dwt = DWT_2D(self.wavelet)
                    self.idwt = IDWT_2D(self.wavelet)
                else:
                    self.dwt = DWT_3D(self.wavelet)
                    self.idwt = IDWT_3D(self.wavelet)

                # Subband normalization to balance energy across subbands
                # This is critical for proper training:
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
                # Order for 3D: [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]
                # Order for 2D: [LL, LH, HL, HH]
                num_subbands = 4 if self.spatial_dims == 2 else 8
                default_weights_3d = [1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0]
                default_weights_2d = [1.0, 2.0, 2.0, 4.0]
                default_weights = default_weights_2d if self.spatial_dims == 2 else default_weights_3d

                subband_loss_weights = p.get("subband_loss_weights", default_weights)
                if subband_loss_weights is not None:
                    self.subband_loss_weights = torch.tensor(subband_loss_weights, dtype=torch.float32)
                    # Normalize weights so they sum to num_subbands
                    self.subband_loss_weights = self.subband_loss_weights * (num_subbands / self.subband_loss_weights.sum())
                    console.log(f"  - Subband loss weights: {self.subband_loss_weights.tolist()}")
                else:
                    self.subband_loss_weights = None
            else:
                self.dwt = None
                self.idwt = None
                self.normalize_subbands = False
                self.subband_normalizer = None
                self.subband_loss_weights = None

            # Wrap UNet for flow matching
            max_timestep = p.get("max_timestep", 1000)
            self.model = WaveletUNetWrapper(
                unet=unet,
                max_timestep=max_timestep,
                apply_wavelet_transform=self.apply_wavelet_transform,
                dwt=self.dwt,
                idwt=self.idwt,
                spatial_dims=self.spatial_dims,
            )

            # Flow matching path
            self.path = AffineProbPath(scheduler=CondOTScheduler())

            # Loss and metrics
            self.mse_loss = MSELoss()
            self.psnr_metric = PSNRMetric(max_val=1.0)
            self.ssim_metric = SSIMMetric(spatial_dims=self.spatial_dims, data_range=1.0)

            # Store configs
            self.solver_args = p.get("solver_args", {})
            self.manual_accumulate_grad_batches = p.get("manual_accumulate_grad_batches", 4)
            self._logging = p.get("_logging", True)

            # Data preprocessing settings
            self.data_range = p.get("data_range", "[-1,1]")

            # Validation generation settings
            self.generate_validation_samples = p.get("generate_validation_samples", True)
            self.generate_frequency = p.get("generate_frequency", 10)
            self.val_max_batches = p.get("val_max_batches", 2)
            self.val_time_points = p.get("val_time_points", 10)

            # Store optimizer and scheduler configs
            self.optim_cfg = mcfg.optim
            self.scheduler_cfg = mcfg.scheduler

            # Note: EMA is handled by EMACallback (configured in callbacks.ema)
            # The callback will set self.ema_model at training start if enabled

            self.save_hyperparameters(ignore=["model", "dwt", "idwt", "subband_normalizer"])

            console.log("[bold green]Wavelet Flow Matching initialized:[/bold green]")
            console.log(f"  - Spatial dims: {self.spatial_dims}")
            console.log(f"  - Wavelet: {self.wavelet}")
            console.log(f"  - Apply wavelet transform: {self.apply_wavelet_transform}")
            console.log(f"  - Normalize subbands: {self.normalize_subbands}")
            console.log(f"  - Solver: {self.solver_args.get('method', 'midpoint')}")

        def _get_inference_model(self):
            """Get the model to use for inference (EMA if available, else main model).

            The EMA model is set by EMACallback during on_fit_start if callbacks.ema
            is configured.
            
            IMPORTANT: We return ema_model.module (not ema_model itself) because:
            - AveragedModel.forward() has a different signature than the wrapped model
            - The .module attribute contains the averaged weights with the original forward()
            """
            ema = getattr(self, "ema_model", None)
            if ema is not None:
                # Return the underlying module with averaged weights
                # AveragedModel.module has the same forward() signature as the original
                return ema.module
            return self.model

        def _normalize_data(self, x: torch.Tensor) -> torch.Tensor:
            """Normalize data to the specified range."""
            if self.data_range == "[-1,1]":
                # Assume input is in [0, 1], scale to [-1, 1]
                return x * 2.0 - 1.0
            elif self.data_range == "[0,1]":
                return x
            else:
                raise ValueError(f"Unknown data_range: {self.data_range}")

        def _denormalize_data(self, x: torch.Tensor) -> torch.Tensor:
            """Denormalize data back to [0, 1] range."""
            if self.data_range == "[-1,1]":
                # Scale from [-1, 1] back to [0, 1]
                return (x + 1.0) / 2.0
            elif self.data_range == "[0,1]":
                return x
            else:
                raise ValueError(f"Unknown data_range: {self.data_range}")
        
        def _apply_dwt(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
            """Apply discrete wavelet transform and stack subbands.
            
            Args:
                x: Input tensor [B, C, ...]
                update_stats: Whether to update normalizer running stats (False for validation)
            """
            if not self.apply_wavelet_transform:
                return x

            subbands = self.dwt(x)
            # Stack subbands along channel dimension
            stacked = torch.cat(subbands, dim=1)

            # Apply subband normalization if enabled
            if self.subband_normalizer is not None:
                stacked = self.subband_normalizer.normalize(stacked, update_stats=update_stats)

            return stacked

        def _apply_idwt(self, x: torch.Tensor) -> torch.Tensor:
            """Split stacked subbands and apply inverse wavelet transform."""
            if not self.apply_wavelet_transform:
                return x

            # Denormalize before IDWT if normalization is enabled
            if self.subband_normalizer is not None:
                x = self.subband_normalizer.denormalize(x)

            num_subbands = 4 if self.spatial_dims == 2 else 8
            subbands = torch.chunk(x, num_subbands, dim=1)
            return self.idwt(*subbands)

        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
            return self.model(x=x, t=t, **extras)

        def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            """Training step with flow matching velocity prediction."""
            images = batch["image"]

            # Normalize data to specified range
            images = self._normalize_data(images)
            
            # Apply DWT to move to wavelet space
            if self.apply_wavelet_transform:
                images_wavelet = self._apply_dwt(images)
            else:
                images_wavelet = images

            # Get optimizer
            opt = self.optimizers()

            # Determine if we should step the optimizer
            is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
            should_step_optimizer = (
                (batch_idx + 1) % self.manual_accumulate_grad_batches == 0
            ) or is_last_batch

            # Sample noise and time IN WAVELET SPACE
            source_dist = torch.randn_like(images_wavelet)
            t = torch.rand(images.shape[0], device=self.device)
            sample_info = self.path.sample(t=t, x_0=source_dist, x_1=images_wavelet)

            # Predict velocity IN WAVELET SPACE
            velocity_pred = self.forward(x=sample_info.x_t, t=sample_info.t)

            # Compute loss - use per-subband weighting if enabled
            if self.subband_loss_weights is not None and self.apply_wavelet_transform:
                # Compute weighted per-subband MSE loss on velocity
                num_subbands = 4 if self.spatial_dims == 2 else 8
                weights = self.subband_loss_weights.to(self.device)

                loss = 0.0
                for i in range(num_subbands):
                    subband_pred = velocity_pred[:, i : i + 1]
                    subband_target = sample_info.dx_t[:, i : i + 1]
                    subband_mse = F.mse_loss(subband_pred, subband_target)
                    loss = loss + weights[i] * subband_mse

                # Average across subbands (weights already normalized to sum to num_subbands)
                loss = loss / num_subbands
            else:
                loss = self.mse_loss(velocity_pred, sample_info.dx_t)

            # Additional monitoring metrics
            if self._logging:
                # Velocity magnitude to detect model collapse/explosion
                velocity_magnitude = torch.norm(
                    velocity_pred.reshape(velocity_pred.shape[0], -1), dim=1
                ).mean()
                self.log("train/velocity_mag", velocity_magnitude, prog_bar=False, sync_dist=True)

                # Time-stratified losses
                early_mask = t < 0.33
                mid_mask = (t >= 0.33) & (t < 0.67)
                late_mask = t >= 0.67

                loss_per_sample = torch.nn.functional.mse_loss(
                    velocity_pred, sample_info.dx_t, reduction="none"
                )
                loss_per_sample = loss_per_sample.reshape(loss_per_sample.shape[0], -1).mean(dim=1)

                # Always log to avoid DDP hangs
                early_loss = (
                    loss_per_sample[early_mask].mean()
                    if early_mask.any()
                    else torch.tensor(0.0, device=self.device)
                )
                mid_loss = (
                    loss_per_sample[mid_mask].mean()
                    if mid_mask.any()
                    else torch.tensor(0.0, device=self.device)
                )
                late_loss = (
                    loss_per_sample[late_mask].mean()
                    if late_mask.any()
                    else torch.tensor(0.0, device=self.device)
                )

                self.log("train/loss_early_t", early_loss, prog_bar=False, sync_dist=True)
                self.log("train/loss_mid_t", mid_loss, prog_bar=False, sync_dist=True)
                self.log("train/loss_late_t", late_loss, prog_bar=False, sync_dist=True)

                # Log per-subband loss periodically (every 50 batches)
                if batch_idx % 50 == 0 and self.apply_wavelet_transform:
                    num_subbands = 4 if self.spatial_dims == 2 else 8
                    subband_names = (
                        ["LL", "LH", "HL", "HH"]
                        if num_subbands == 4
                        else ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                    )
                    for i, name in enumerate(subband_names):
                        subband_pred = velocity_pred[:, i : i + 1]
                        subband_target = sample_info.dx_t[:, i : i + 1]
                        subband_loss = F.mse_loss(subband_pred, subband_target)
                        self.log(
                            f"train/subband_loss/{name}",
                            subband_loss,
                            prog_bar=False,
                            sync_dist=True,
                        )

                    # Also log subband normalizer stats if enabled
                    if self.subband_normalizer is not None:
                        for i, name in enumerate(subband_names):
                            self.log(
                                f"subband_std/{name}",
                                self.subband_normalizer.running_std[i],
                                prog_bar=False,
                                sync_dist=True,
                            )

            # Scale loss for gradient accumulation
            loss = loss / self.manual_accumulate_grad_batches

            # Manual backward
            self.manual_backward(loss)

            # Step optimizer if accumulated enough gradients
            if should_step_optimizer:
                opt.step()
                opt.zero_grad(set_to_none=True)  # Zero gradients AFTER stepping
                # Note: EMA update is handled by EMACallback.on_train_batch_end()

            if self._logging:
                # Log the unscaled loss for monitoring
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
            # Cache first batch for sample generation
            if batch_idx == 0:
                self._cached_val_batch = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }

            images = batch["image"]

            # Normalize data to specified range
            images = self._normalize_data(images)
            
            # Apply DWT to move to wavelet space (don't update normalizer stats during validation)
            if self.apply_wavelet_transform:
                images_wavelet = self._apply_dwt(images, update_stats=False)
            else:
                images_wavelet = images

            source_dist = torch.randn_like(images_wavelet)
            t = torch.rand(images.shape[0], device=self.device)
            sample_info = self.path.sample(t=t, x_0=source_dist, x_1=images_wavelet)

            # Use EMA model for validation if available
            model_to_use = self._get_inference_model()
            velocity_pred = model_to_use(x=sample_info.x_t, t=sample_info.t)

            loss = self.mse_loss(velocity_pred, sample_info.dx_t)

            if self._logging:
                self.log("val/loss", loss, prog_bar=True, sync_dist=True)

                # Log per-subband validation loss
                if self.apply_wavelet_transform:
                    num_subbands = 4 if self.spatial_dims == 2 else 8
                    subband_names = (
                        ["LL", "LH", "HL", "HH"]
                        if num_subbands == 4
                        else ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                    )
                    for i, name in enumerate(subband_names):
                        subband_pred = velocity_pred[:, i : i + 1]
                        subband_target = sample_info.dx_t[:, i : i + 1]
                        subband_loss = F.mse_loss(subband_pred, subband_target)
                        self.log(
                            f"val/subband_loss/{name}",
                            subband_loss,
                            prog_bar=False,
                            sync_dist=True,
                        )

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
                self._compute_generation_metrics(
                    max_batches=self.val_max_batches,
                    time_points=self.val_time_points,
                )

        def _compute_generation_metrics(self, max_batches: int = 2, time_points: int = 10):
            """Lightweight metric computation with configurable sampling.

            Args:
                max_batches: Number of validation batches to evaluate
                time_points: Number of ODE solver steps (fewer = faster)
            """
            # Use EMA model for generation if available
            model_to_use = self._get_inference_model()
            solver = ODESolver(velocity_model=model_to_use)

            # Use cached validation batch
            if not hasattr(self, "_cached_val_batch") or self._cached_val_batch is None:
                console.log("[yellow]No cached validation batch, skipping generation[/yellow]")
                return

            psnr_scores = []
            ssim_scores = []

            model_to_use.eval()

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            console.log(
                f"[cyan]Generating samples for epoch {self.current_epoch} "
                f"({time_points} ODE steps)...[/cyan]"
            )

            with torch.no_grad():
                imgs_original = self._cached_val_batch["image"].to(self.device)
                imgs = self._normalize_data(imgs_original)
                
                # Apply DWT to get wavelet space representation
                if self.apply_wavelet_transform:
                    imgs_wavelet = self._apply_dwt(imgs)
                else:
                    imgs_wavelet = imgs
                
                # Start from random noise IN WAVELET SPACE
                x_init = torch.randn_like(imgs_wavelet)

                # Generate samples IN WAVELET SPACE
                time_grid = torch.linspace(0, 1, time_points, device=self.device)
                sol = solver.sample(
                    time_grid=time_grid,
                    step_size=None,
                    x_init=x_init,
                    method=self.solver_args.get("method", "midpoint"),
                    return_intermediates=False,
                )

                final_imgs_wavelet = sol
                
                # Apply IDWT to convert back to image space
                if self.apply_wavelet_transform:
                    final_imgs = self._apply_idwt(final_imgs_wavelet)
                else:
                    final_imgs = final_imgs_wavelet

                # Denormalize generated images back to [0, 1] for metrics
                final_imgs_denorm = self._denormalize_data(final_imgs)
                final_imgs_denorm = torch.clamp(final_imgs_denorm, 0.0, 1.0)

                # Compute metrics
                psnr = self.psnr_metric(final_imgs_denorm, imgs_original).mean()
                ssim = self.ssim_metric(final_imgs_denorm, imgs_original).mean()

                psnr_scores.append(psnr.item() if isinstance(psnr, torch.Tensor) else psnr)
                ssim_scores.append(ssim.item() if isinstance(ssim, torch.Tensor) else ssim)

                # Save visualization
                self._save_generated_center_slice(
                    final_imgs_denorm[0], imgs_original[0], self.current_epoch
                )

            # Log average metrics
            if psnr_scores:
                avg_psnr = sum(psnr_scores) / len(psnr_scores)
                avg_ssim = sum(ssim_scores) / len(ssim_scores)
                self.log("val/psnr", avg_psnr, prog_bar=True, sync_dist=False, rank_zero_only=True)
                self.log("val/ssim", avg_ssim, prog_bar=True, sync_dist=False, rank_zero_only=True)

                console.log(
                    f"[bold green]✓[/bold green] Epoch {self.current_epoch} - "
                    f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f} "
                    f"({time_points} ODE steps)"
                )

        def _save_generated_center_slice(
            self, gen_vol: torch.Tensor, orig_vol: torch.Tensor, epoch: int
        ):
            """Save generated and original center slices for visual monitoring."""
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            # Get log directory
            log_dir = self._get_log_dir()
            samples_dir = os.path.join(log_dir, "wavelet_flow_matching_samples")
            os.makedirs(samples_dir, exist_ok=True)

            gen = gen_vol.cpu().squeeze()
            orig = orig_vol.cpu().squeeze()

            # Handle 3D: take center slice
            if gen.dim() == 3:
                center = gen.shape[0] // 2
                gen = gen[center]
                orig = orig[center]

            # Normalize for display
            gen = (gen - gen.min()) / (gen.max() - gen.min() + 1e-8)
            orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)

            # Rotate for correct orientation
            gen = np.rot90(gen.numpy(), k=-1)
            orig = np.rot90(orig.numpy(), k=-1)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(gen, cmap="gray")
            axes[1].set_title("Generated")
            axes[1].axis("off")

            plt.suptitle(f"Epoch {epoch}")
            plt.tight_layout()
            save_path = os.path.join(samples_dir, f"epoch_{epoch:04d}_sample.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close(fig)

            # Log to wandb if available
            if self.logger is not None and hasattr(self.logger, "experiment"):
                try:
                    import wandb

                    comparison = np.concatenate([orig, gen], axis=1)
                    self.logger.experiment.log(
                        {
                            "generated_samples": wandb.Image(comparison, caption=f"Epoch {epoch}"),
                            "epoch": epoch,
                        }
                    )
                except Exception as e:
                    console.log(f"[yellow]Could not log images to W&B: {e}[/yellow]")

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
            save_comparison = getattr(self, "inference_save_comparison", True) and not is_unconditional
            
            # Get batch data
            imgs_original = batch["image"].to(device)
            batch_size = imgs_original.shape[0]

            # Normalize images to model's expected range
            imgs_normalized = self._normalize_data(imgs_original)
            
            # Apply DWT to get wavelet space representation
            if self.apply_wavelet_transform:
                imgs_wavelet = self._apply_dwt(imgs_normalized)
            else:
                imgs_wavelet = imgs_normalized

            # Generate samples from random noise IN WAVELET SPACE
            x_init = torch.randn_like(imgs_wavelet)

            # Solver configuration
            time_points = self.solver_args.get("time_points", 20)
            method = self.solver_args.get("method", "midpoint")

            # Use EMA model for generation if available
            model_to_use = self._get_inference_model()
            model_to_use.eval()

            # Create ODE solver and generate samples IN WAVELET SPACE
            solver = ODESolver(velocity_model=model_to_use)
            time_grid = torch.linspace(0, 1, time_points, device=device)

            with torch.no_grad():
                generated_wavelet = solver.sample(
                    time_grid=time_grid,
                    step_size=None,
                    x_init=x_init,
                    method=method,
                    return_intermediates=False,
                )
            
            # Apply IDWT to convert back to image space
            if self.apply_wavelet_transform:
                generated_samples = self._apply_idwt(generated_wavelet)
            else:
                generated_samples = generated_wavelet

            # Denormalize generated samples back to [0, 1] range
            generated_samples = self._denormalize_data(generated_samples)
            generated_samples = torch.clamp(generated_samples, 0.0, 1.0)

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
                img = np.rot90(img.numpy(), k=-1)
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Save using PIL at native resolution
                from PIL import Image
                png_path = os.path.join(samples_dir, f"sample_{sample_idx:05d}.png")
                Image.fromarray(img_uint8, mode='L').save(png_path)
                
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

    return LitWaveletFlowMatching(cfg)
