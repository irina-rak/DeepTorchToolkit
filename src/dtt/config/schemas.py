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
    devices: int | str | None = None  # e.g., 1, "auto"
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


class CallbacksConfig(BaseModel):
    model_checkpoint: ModelCheckpointConfig = Field(default_factory=ModelCheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    lr_monitor: LRMonitorConfig = Field(default_factory=LRMonitorConfig)


class Config(BaseModel):
    seed: int = 42
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)

