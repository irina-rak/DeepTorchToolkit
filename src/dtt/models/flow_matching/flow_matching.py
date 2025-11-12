"""Flow Matching model for generative modeling.

This module implements a Flow Matching model using the DTT framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from dtt.utils.registry import register_model

__all__ = ["build_flow_matching"]


@register_model("flow_matching")
def build_flow_matching(cfg: dict[str, Any]):
    """Build Flow Matching model from config.

    Expected config structure:
        model:
            name: "flow_matching"
            optim:
                name: adam | adamw | sgd
                lr: float (e.g., 3e-4)
                weight_decay: float (default: 0.0)
            scheduler:
                name: cosine | reduce_on_plateau | step | exponential | null
                params: {}  # scheduler-specific params
            params:
                unet_config: dict with DiffusionModelUNet params
                    spatial_dims: int (2 or 3)
                    in_channels: int
                    out_channels: int
                    num_res_blocks: list[int]
                    channels: list[int]
                    attention_levels: list[bool]
                    norm_num_groups: int
                    num_head_channels: int | list[int]
                    resblock_updown: bool (default: False)
                    with_conditioning: bool (default: False)
                    transformer_num_layers: int (default: 1)
                    use_flash_attention: bool (default: False)
                    dropout_cattn: float (default: 0.0)
                solver_args: dict
                    time_points: int (default: 20) - ODE steps for training generation
                    method: str (default: "midpoint") - ODE solver method
                max_timestep: int (default: 1000)
                    Maximum timestep value for rescaling t from [0,1] to [0, max_timestep]
                manual_accumulate_grad_batches: int (default: 4)
                seed: int | None (default: None)
                _logging: bool (default: True)
                # Data preprocessing
                data_range: str (default: "[-1,1]")
                    Data normalization range: "[-1,1]" (recommended) or "[0,1]"
                    [-1,1] improves training stability (following Facebook's implementation)
                # EMA (Exponential Moving Average) settings
                use_ema: bool (default: False)
                    Enable EMA for improved generation quality
                ema_decay: float (default: 0.9999)
                    EMA decay rate (higher = slower updates)
                # Validation generation settings (for performance tuning)
                generate_validation_samples: bool (default: True)
                    Whether to generate samples during validation
                generate_frequency: int (default: 5)
                    Generate samples every N epochs (reduces validation overhead)
                val_max_batches: int (default: 3)
                    Number of validation batches to use for generation metrics
                val_time_points: int (default: 10)
                    ODE steps for validation generation (fewer = faster)
    """
    # Lazy imports
    import torch
    from flow_matching.utils.model_wrapper import ModelWrapper
    from lightning.pytorch import LightningModule
    from monai.metrics import PSNRMetric, SSIMMetric
    from monai.networks.nets import DiffusionModelUNet
    from monai.utils import set_determinism
    from torch.nn import MSELoss

    from dtt.models.base import build_optimizer, build_scheduler
    from dtt.utils.logging import get_console

    console = get_console()

    # Enable torch.compile if available
    try:
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass

    class DiffusionUNetWrapper(ModelWrapper):
        """flow_matching library expects a slightly different interface in the forward
        method. This is a wrapper to allow both x and t inputs, along with extras.

        This adapter works as a bridge between:
        - flow_matching's expected interface: forward(x, t, **extras)
        - MONAI's DiffusionModelUNet interface: forward(x, timesteps, context)

        DiffusionModelUNet expects timesteps in a specific range, so this also rescales
        timesteps from [0, 1] (flow_matching convention) to [0, max_timestep] with respect
        to diffusion model convention.
        """

        def __init__(self, unet: DiffusionModelUNet, max_timestep: int = 1000):
            """Initialize the wrapper.

            Args:
                unet: MONAI DiffusionModelUNet instance
                max_timestep: Maximum timestep value for rescaling (default: 1000)
            """
            super().__init__(model=unet)
            self.max_timestep = max_timestep

        def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            **extras,
        ) -> torch.Tensor:
            """Forward pass with interface adaptation and timestep rescaling.

            Args:
                x: Input tensor (batch_size, channels, *spatial_dims)
                t: Timesteps in [0, 1] (batch_size,)
                **extras: Additional arguments:
                    - context/cond: Conditioning tensor (optional)
                    - masks: Mask tensor (optional, not used by DiffusionModelUNet)

            Returns:
                Model output (same shape as x)
            """
            # Rescale t from [0, 1] to [0, max_timestep]
            t_scaled = t * self.max_timestep

            # Ensure timesteps is 1D for MONAI's DiffusionModelUNet
            # Handle scalar timesteps from ODE solver
            if t_scaled.dim() == 0:
                # Scalar timestep - expand to batch size
                batch_size = x.shape[0]
                t_scaled = t_scaled.expand(batch_size)
            elif t_scaled.dim() > 1:
                # Multi-dimensional - flatten to 1D
                t_scaled = t_scaled.flatten()

            # Extract conditioning if provided (supports both 'context' and 'cond' keys)
            context = extras.get("context") or extras.get("cond")

            # Call MONAI's DiffusionModelUNet
            output = self.model(x=x, timesteps=t_scaled, context=context)
            return output

    class LitFlowMatching(LightningModule):
        """Lightning module for Flow Matching."""

        def __init__(self, config: Any):
            super().__init__()

            # Normalize config to Config object
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

            # Extract UNet configuration
            unet_config = p.get("unet_config", {})
            max_timestep = p.get("max_timestep", 1000)

            # Initialize UNet wrapped for flow_matching compatibility
            unet = DiffusionModelUNet(**unet_config)
            self.model = DiffusionUNetWrapper(unet=unet, max_timestep=max_timestep)

            # Solver - lazy import
            from flow_matching.path import AffineProbPath
            from flow_matching.path.scheduler import CondOTScheduler

            self.path = AffineProbPath(scheduler=CondOTScheduler())

            # Loss and metrics
            self.mse_loss = MSELoss()
            self.psnr_metric = PSNRMetric(max_val=1.0)
            spatial_dims = unet_config.get("spatial_dims", 2)
            self.ssim_metric = SSIMMetric(spatial_dims=spatial_dims, data_range=1.0)

            # Store configs from model params
            self.solver_args = p.get("solver_args", {})
            self.manual_accumulate_grad_batches = p.get("manual_accumulate_grad_batches", 4)
            self._logging = p.get("_logging", True)

            # Data preprocessing settings
            self.data_range = p.get("data_range", "[-1,1]")  # "[-1,1]" or "[0,1]"

            # Validation generation settings
            self.generate_validation_samples = p.get("generate_validation_samples", True)
            self.generate_frequency = p.get("generate_frequency", 5)  # Every 5 epochs by default
            self.val_max_batches = p.get("val_max_batches", 3)
            self.val_time_points = p.get("val_time_points", 10)  # Fewer steps for validation

            # EMA settings
            self.use_ema = p.get("use_ema", False)
            self.ema_decay = p.get("ema_decay", 0.9999)
            if self.use_ema:
                self.model_ema = self._create_ema_model()
                console.log(f"[bold cyan]EMA enabled with decay={self.ema_decay}[/bold cyan]")

            # Store optimizer and scheduler configs from model config
            self.optim_cfg = mcfg.optim
            self.scheduler_cfg = mcfg.scheduler

            self.save_hyperparameters(ignore=["model", "model_ema"])

        def _create_ema_model(self):
            """Create EMA (Exponential Moving Average) copy of the model."""
            import copy

            ema_model = copy.deepcopy(self.model)
            ema_model.eval()
            ema_model.requires_grad_(False)
            return ema_model

        def _update_ema(self):
            """Update EMA model with exponential moving average of current model parameters."""
            if not self.use_ema:
                return

            with torch.no_grad():
                for ema_param, model_param in zip(
                    self.model_ema.parameters(), self.model.parameters(), strict=True
                ):
                    ema_param.data.mul_(self.ema_decay).add_(
                        model_param.data, alpha=1 - self.ema_decay
                    )

        def _normalize_data(self, x: torch.Tensor) -> torch.Tensor:
            """Normalize data to the specified range."""
            if self.data_range == "[-1,1]":
                # Assume input is in [0, 1], scale to [-1, 1]
                return x * 2.0 - 1.0
            elif self.data_range == "[0,1]":
                # Keep as is
                return x
            else:
                raise ValueError(f"Unknown data_range: {self.data_range}")

        def _denormalize_data(self, x: torch.Tensor) -> torch.Tensor:
            """Denormalize data back to [0, 1] range."""
            if self.data_range == "[-1,1]":
                # Scale from [-1, 1] back to [0, 1]
                return (x + 1.0) / 2.0
            elif self.data_range == "[0,1]":
                # Keep as is
                return x
            else:
                raise ValueError(f"Unknown data_range: {self.data_range}")

        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
            return self.model(x=x, t=t, **extras)

        def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            """Training step with manual gradient accumulation."""
            images = batch["image"]

            # Normalize data to specified range
            images = self._normalize_data(images)

            # Get optimizer
            opt = self.optimizers()

            # Determine if we should step the optimizer
            is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
            should_step_optimizer = (
                (batch_idx + 1) % self.manual_accumulate_grad_batches == 0
            ) or is_last_batch

            # Zero gradients if we're about to step
            if should_step_optimizer:
                opt.zero_grad(set_to_none=True)

            # Sample noise and time
            source_dist = torch.randn_like(images)
            t = torch.rand(images.shape[0], device=self.device)
            sample_info = self.path.sample(t=t, x_0=source_dist, x_1=images)

            # Predict velocity
            velocity_pred = self.forward(x=sample_info.x_t, t=sample_info.t)

            # Compute loss
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

                if early_mask.any():
                    self.log(
                        "train/loss_early_t",
                        loss_per_sample[early_mask].mean(),
                        prog_bar=False,
                        sync_dist=True,
                    )
                if mid_mask.any():
                    self.log(
                        "train/loss_mid_t",
                        loss_per_sample[mid_mask].mean(),
                        prog_bar=False,
                        sync_dist=True,
                    )
                if late_mask.any():
                    self.log(
                        "train/loss_late_t",
                        loss_per_sample[late_mask].mean(),
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
                # Update EMA after optimizer step
                if self.use_ema:
                    self._update_ema()

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
            images = batch["image"]

            # Normalize data to specified range
            images = self._normalize_data(images)

            source_dist = torch.randn_like(images)
            t = torch.rand(images.shape[0], device=self.device)
            sample_info = self.path.sample(t=t, x_0=source_dist, x_1=images)

            # Use EMA model for validation if available
            model_to_use = self.model_ema if self.use_ema else self.model
            velocity_pred = model_to_use(x=sample_info.x_t, t=sample_info.t)

            loss = self.mse_loss(velocity_pred, sample_info.dx_t)

            if self._logging:
                self.log("val/loss", loss, prog_bar=True, sync_dist=True)

            return {"loss": loss}

        def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            """Test step not implemented."""
            raise NotImplementedError("Test step is not implemented for LitFlowMatching.")

        def on_validation_epoch_end(self) -> None:
            """Run lightweight sampling for metrics and visualization.

            Generation is configurable and runs only every N epochs to avoid
            excessive computational overhead during training.
            """
            # Avoid duplicate work under DDP
            if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
                return

            # Check if generation is enabled and if it's time to generate
            should_generate = (
                self.generate_validation_samples
                and self.current_epoch % self.generate_frequency == 0
            )

            if not should_generate:
                return

            # Lightweight validation: compute metrics and save center slices only
            self._compute_generation_metrics(
                max_batches=self.val_max_batches,
                time_points=self.val_time_points,
            )

        def _compute_generation_metrics(self, max_batches: int = 3, time_points: int = 10):
            """Lightweight metric computation with configurable sampling.

            Args:
                max_batches: Number of validation batches to evaluate
                time_points: Number of ODE solver steps (fewer = faster)
            """
            from flow_matching.solver import ODESolver

            # Use EMA model for generation if available
            model_to_use = self.model_ema if self.use_ema else self.model
            solver = ODESolver(velocity_model=model_to_use)
            val_loader = self.trainer.datamodule.val_dataloader()

            psnr_scores = []
            ssim_scores = []

            model_to_use.eval()

            # Only use progress bar on rank 0 to avoid file handle issues in DDP
            use_progress = (
                not hasattr(self.trainer, "is_global_zero") or self.trainer.is_global_zero
            )

            if use_progress:
                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )

                progress_ctx = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=console,
                )
            else:
                from contextlib import nullcontext

                progress_ctx = nullcontext()

            with progress_ctx as progress:
                if use_progress:
                    task = progress.add_task(
                        f"[cyan]Computing generation metrics (epoch {self.current_epoch})...",
                        total=max_batches,
                    )

                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        if i >= max_batches:
                            break

                        imgs_original = batch["image"].to(self.device)
                        # Normalize for generation
                        imgs = self._normalize_data(imgs_original)
                        x_init = torch.randn_like(imgs)

                        # Generate samples with fewer time points for validation
                        time_grid = torch.linspace(0, 1, time_points, device=self.device)
                        sol = solver.sample(
                            time_grid=time_grid,
                            step_size=None,
                            x_init=x_init,
                            method=self.solver_args.get("method", "midpoint"),
                            return_intermediates=False,  # Don't store trajectory
                        )

                        final_imgs = sol

                        # Denormalize generated images back to [0, 1] for metrics
                        final_imgs_denorm = self._denormalize_data(final_imgs)
                        final_imgs_denorm = torch.clamp(final_imgs_denorm, 0.0, 1.0)

                        # Compute metrics (PSNR/SSIM expect [0, 1] range)
                        psnr = self.psnr_metric(final_imgs_denorm, imgs_original).mean()
                        ssim = self.ssim_metric(final_imgs_denorm, imgs_original).mean()

                        psnr_scores.append(psnr.item() if isinstance(psnr, torch.Tensor) else psnr)
                        ssim_scores.append(ssim.item() if isinstance(ssim, torch.Tensor) else ssim)

                        # Save quick center slice visualization for first batch only (rank 0 only)
                        if i == 0 and use_progress:
                            self._save_center_slice_comparison(
                                imgs_original[0], final_imgs_denorm[0], self.current_epoch
                            )

                        if use_progress:
                            progress.update(task, advance=1)

            # Log average metrics
            if psnr_scores:
                avg_psnr = sum(psnr_scores) / len(psnr_scores)
                avg_ssim = sum(ssim_scores) / len(ssim_scores)
                self.log("val/psnr", avg_psnr, prog_bar=True, sync_dist=True)
                self.log("val/ssim", avg_ssim, prog_bar=True, sync_dist=True)

                if use_progress:
                    console.log(
                        f"[bold green]✓[/bold green] Epoch {self.current_epoch} - "
                        f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f} "
                        f"({time_points} ODE steps)"
                    )

        def _save_center_slice_comparison(
            self, real_vol: torch.Tensor, gen_vol: torch.Tensor, epoch: int
        ):
            """Save a quick center slice comparison for visual monitoring."""
            import os

            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend
            import matplotlib.pyplot as plt

            # Try to get log directory from various sources
            log_dir = None

            # First, try logger's log_dir (works for WandB, TensorBoard, etc.)
            if self.trainer.logger is not None:
                # For WandB, check experiment.dir first (most reliable)
                if hasattr(self.trainer.logger, "experiment"):
                    if hasattr(self.trainer.logger.experiment, "dir"):
                        log_dir = self.trainer.logger.experiment.dir
                        console.log(f"[dim]Using WandB experiment.dir: {log_dir}[/dim]")
                    elif hasattr(self.trainer.logger.experiment, "path"):
                        log_dir = str(self.trainer.logger.experiment.path)
                        console.log(f"[dim]Using WandB experiment.path: {log_dir}[/dim]")
                
                # Fallback to logger attributes
                if not log_dir:
                    if hasattr(self.trainer.logger, "save_dir") and self.trainer.logger.save_dir:
                        log_dir = self.trainer.logger.save_dir
                        console.log(f"[dim]Using logger.save_dir: {log_dir}[/dim]")
                    elif hasattr(self.trainer.logger, "log_dir") and self.trainer.logger.log_dir:
                        log_dir = self.trainer.logger.log_dir
                        console.log(f"[dim]Using logger.log_dir: {log_dir}[/dim]")

            # Fallback to trainer's default_root_dir with absolute path
            if not log_dir or log_dir == ".":
                log_dir = os.path.abspath(self.trainer.default_root_dir)
                console.log(f"[dim]Using trainer.default_root_dir: {log_dir}[/dim]")

            # Create quick_samples subdirectory
            samples_dir = os.path.join(log_dir, "quick_samples")
            os.makedirs(samples_dir, exist_ok=True)

            # Get center slices from 3D volume [C, D, H, W]
            real = real_vol.cpu().squeeze()  # [D, H, W] or [H, W]
            gen = gen_vol.cpu().squeeze()

            if real.dim() == 3:  # 3D volume
                center_slice = real.shape[0] // 2
                real_slice = real[center_slice]
                gen_slice = gen[center_slice]
            else:  # 2D image
                real_slice = real
                gen_slice = gen

            # Normalize to [0, 1] for display
            real_slice = (real_slice - real_slice.min()) / (
                real_slice.max() - real_slice.min() + 1e-8
            )
            gen_slice = (gen_slice - gen_slice.min()) / (gen_slice.max() - gen_slice.min() + 1e-8)

            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(gen_slice.numpy(), cmap="gray")
            axes[0].set_title(f"Generated (Epoch {epoch})")
            axes[0].axis("off")

            axes[1].imshow(real_slice.numpy(), cmap="gray")
            axes[1].set_title("Real")
            axes[1].axis("off")

            plt.tight_layout()
            save_path = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            
            # Log to WandB if available
            if self.trainer.logger is not None:
                try:
                    import wandb
                    
                    # Check if this is a WandB logger
                    if hasattr(self.trainer.logger, "experiment") and isinstance(
                        self.trainer.logger.experiment, (wandb.sdk.wandb_run.Run, wandb.wandb_run.Run)
                    ):
                        # Log the image to WandB
                        self.trainer.logger.experiment.log(
                            {
                                "samples/comparison": wandb.Image(
                                    save_path,
                                    caption=f"Epoch {epoch}: Generated vs Real",
                                )
                            },
                            step=self.global_step,
                        )
                except ImportError:
                    pass  # WandB not installed, skip logging
                except Exception as e:
                    console.log(f"[yellow]Warning: Could not log image to WandB: {e}[/yellow]")
            
            plt.close(fig)  # Explicitly close the figure
            plt.close("all")  # Close all figures to be safe

            # Clear matplotlib cache
            matplotlib.pyplot.clf()
            matplotlib.pyplot.cla()

            console.log(f"[dim]Saved sample comparison to: {save_path}[/dim]")

        def configure_optimizers(self):
            """Configure optimizer and scheduler."""
            optimizer = build_optimizer(self.model.parameters(), self.optim_cfg)

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

    return LitFlowMatching(cfg)
