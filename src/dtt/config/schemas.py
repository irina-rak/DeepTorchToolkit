from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OptimConfig(BaseModel):
    name: str = Field(default="adam")
    lr: float = Field(default=1e-3)
    weight_decay: float = Field(default=0.0)


class SchedulerConfig(BaseModel):
    name: str | None = None  # e.g., "cosine", "reduce_on_plateau", "step"
    params: dict[str, Any] = Field(default_factory=dict)  # scheduler-specific params


class TrainerConfig(BaseModel):
    max_epochs: int = 1
    accelerator: str = "auto"
    strategy: str | list[str] | None = None  # e.g., "ddp", ["ddp_find_unused_parameters_false"]
    devices: int | list[int] | str | None = None  # e.g., 1, "auto"
    precision: str | int = "32-true"  # lightning style string acceptable
    log_every_n_steps: int = 50


class ModelConfig(BaseModel):
    name: str = Field(default="monai.unet")
    params: dict[str, Any] = Field(default_factory=dict)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    metrics: list[str] = Field(default_factory=list)  # e.g., ["dice", "hausdorff"]


class DataConfig(BaseModel):
    name: str = Field(default="medical2d")
    params: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = 4
    num_workers: int = 4


class WandBConfig(BaseModel):
    project: str = "dtt"
    name: str | None = None
    entity: str | None = None
    tags: list[str] = Field(default_factory=list)
    mode: str = "offline"  # offline by default for safety
    api_key: str | None = None  # Optional: override WANDB_API_KEY env var


class LoggerConfig(BaseModel):
    wandb: WandBConfig = Field(default_factory=WandBConfig)


class ModelCheckpointConfig(BaseModel):
    monitor: str = "val/loss"
    save_top_k: int = 1
    mode: str = "min"
    filename: str = "epoch{epoch:02d}-valloss{val/loss:.3f}"
    dirpath: str | None = None
    save_last: bool = True
    verbose: bool = False


class EarlyStoppingConfig(BaseModel):
    monitor: str = "val/loss"
    patience: int = 10
    mode: str = "min"
    min_delta: float = 0.0
    verbose: bool = False


class LRMonitorConfig(BaseModel):
    logging_interval: str = "epoch"
    log_momentum: bool = False


class EMAConfig(BaseModel):
    """Configuration for EMA (Exponential Moving Average) callback.

    EMA maintains a moving average of model weights for more stable inference.
    Using swa_utils.AveragedModel ensures both parameters AND buffers are averaged.
    """

    decay: float = 0.9999  # EMA decay rate (higher = slower update, more stable)
    update_every: int = 10  # Update every N steps (saves compute with minimal quality impact)
    update_bn_on_train_end: bool = True  # Refresh BatchNorm stats after training


class CallbacksConfig(BaseModel):
    model_checkpoint: ModelCheckpointConfig = Field(default_factory=ModelCheckpointConfig)
    early_stopping: EarlyStoppingConfig | None = None  # Optional - set to None to disable
    lr_monitor: LRMonitorConfig = Field(default_factory=LRMonitorConfig)
    ema: EMAConfig | None = None  # Optional - set to enable EMA callback


class InferenceConfig(BaseModel):
    use_test_data: bool = True  # If False, uses dummy data for unconditional generation
    num_batches: int = 10  # Number of batches for unconditional generation


class EvaluationConfig(BaseModel):
    """Configuration for evaluation of generated images.

    This config is used by the `dtt evaluate` command to compute
    distribution-based metrics (FID/KID) between real and generated images.
    """

    real_dir: str  # Directory or JSON file containing real images
    fake_dir: str  # Directory or JSON file containing generated/fake images
    spatial_dims: int = 2  # 2 for images, 3 for volumes
    feature_extractor: str = "auto"  # auto, inception, medicalnet
    max_samples: int | None = None  # Maximum samples to load (None = all)
    batch_size: int = 32  # Batch size for feature extraction
    device: str = "cuda"  # Device for computation
    compute_kid: bool = True  # Whether to compute KID in addition to FID
    output_path: str | None = None  # Path to save results as JSON

    # Image resizing
    target_size: list[int] | None = None  # Target size to resize all images (e.g., [128, 128])

    # KID-specific parameters
    kid_num_subsets: int = 100  # Number of subsets for KID computation
    kid_subset_size: int = 1000  # Size of each subset

    # MedicalNet-specific (for 3D)
    medicalnet_depth: int = 50  # ResNet depth: 10, 18, 34, or 50

    # Visualization options
    save_visualizations: bool = False  # Whether to generate visualization plots
    visualization_dir: str | None = None  # Directory for plots (defaults to output_path parent)
    plot_tsne: bool = True  # t-SNE scatter plot of features
    plot_sample_grid: bool = True  # Side-by-side sample comparison
    plot_histogram: bool = True  # Pixel intensity distributions
    num_grid_samples: int = 16  # Number of samples per side in grid


class Config(BaseModel):
    seed: int = 42
    save_dir: str = "experiments"  # Base directory for all training outputs
    ckpt_path: str | None = None  # Path to checkpoint for resuming training
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)  # For inference configs
    evaluation: EvaluationConfig | None = None  # For evaluation configs
