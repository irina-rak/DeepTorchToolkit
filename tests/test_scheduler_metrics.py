from __future__ import annotations

import pytest


def test_scheduler_config_validation():
    """Test that scheduler configs validate correctly."""
    from dtt.config.schemas import Config

    # Cosine scheduler
    cfg_dict = {
        "seed": 1,
        "trainer": {"max_epochs": 1},
        "model": {"scheduler": {"name": "cosine", "params": {"T_max": 50, "eta_min": 1e-6}}},
    }
    cfg = Config.model_validate(cfg_dict)
    assert cfg.model.scheduler.name == "cosine"
    assert cfg.model.scheduler.params["T_max"] == 50

    # No scheduler
    cfg_dict_no_sched = {
        "seed": 1,
        "trainer": {"max_epochs": 1},
        "model": {"scheduler": {"name": None}},
    }
    cfg = Config.model_validate(cfg_dict_no_sched)
    assert cfg.model.scheduler.name is None


def test_metrics_config_validation():
    """Test that metrics list validates correctly."""
    from dtt.config.schemas import Config

    cfg_dict = {
        "seed": 1,
        "trainer": {"max_epochs": 1},
        "model": {"metrics": ["dice", "hausdorff"]},
    }
    cfg = Config.model_validate(cfg_dict)
    assert "dice" in cfg.model.metrics
    assert "hausdorff" in cfg.model.metrics


def test_callback_schemas_validated():
    """Test that callback configs are validated via Pydantic."""
    from dtt.config.schemas import Config

    # Valid callback config
    cfg_dict = {
        "seed": 1,
        "callbacks": {
            "model_checkpoint": {"monitor": "val/dice", "mode": "max", "save_top_k": 3},
            "early_stopping": {"patience": 20, "min_delta": 0.001},
        },
    }
    cfg = Config.model_validate(cfg_dict)
    assert cfg.callbacks.model_checkpoint.monitor == "val/dice"
    assert cfg.callbacks.model_checkpoint.mode == "max"
    assert cfg.callbacks.early_stopping.patience == 20

    # Note: Pydantic accepts arbitrary strings for mode field
    # Could add enum constraint if stricter validation needed


@pytest.mark.heavy
def test_scheduler_builder():
    """Test that schedulers are built correctly (requires torch)."""
    try:
        import torch
    except ImportError:
        pytest.skip("Torch not installed")

    from dtt.config.schemas import OptimConfig, SchedulerConfig
    from dtt.models.base import build_optimizer, build_scheduler

    # Create dummy parameters
    params = [torch.nn.Parameter(torch.randn(2, 2))]

    optim_cfg = OptimConfig(name="adam", lr=1e-3)
    optimizer = build_optimizer(params, optim_cfg)

    # Test cosine scheduler
    sched_cfg = SchedulerConfig(name="cosine", params={"T_max": 10})
    scheduler = build_scheduler(optimizer, sched_cfg)
    assert scheduler is not None
    assert hasattr(scheduler, "step")

    # Test no scheduler
    sched_cfg_none = SchedulerConfig(name=None)
    scheduler_none = build_scheduler(optimizer, sched_cfg_none)
    assert scheduler_none is None


@pytest.mark.heavy
def test_example_config_with_scheduler():
    """Test that example config with scheduler validates and works."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("Torch not installed")

    from dtt.config.schemas import Config

    example_cfg = {
        "seed": 42,
        "trainer": {"max_epochs": 5},
        "model": {
            "name": "monai.unet",
            "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01},
            "scheduler": {"name": "cosine", "params": {"T_max": 50}},
            "metrics": ["dice"],
        },
    }
    cfg = Config.model_validate(example_cfg)
    assert cfg.model.scheduler.name == "cosine"
    assert cfg.model.metrics == ["dice"]
