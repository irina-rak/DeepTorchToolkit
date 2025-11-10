from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from dtt.config.schemas import Config
from dtt.utils.io import read_yaml, save_yaml


def test_config_merging():
    """Test that user config properly overrides defaults."""
    # Create a minimal override config
    override = {
        "seed": 999,
        "trainer": {"max_epochs": 10},
        "model": {"name": "custom_model", "optim": {"lr": 0.001}},
    }

    # Simulate merging (like CLI does)
    from importlib.resources import files

    defaults_path = files("dtt.config").joinpath("defaults.yaml")
    base_cfg = read_yaml(defaults_path)
    base_cfg.update(override)

    # Validate merged config
    cfg = Config.model_validate(base_cfg)

    # Assert overrides took effect
    assert cfg.seed == 999
    assert cfg.trainer.max_epochs == 10
    assert cfg.model.name == "custom_model"
    assert cfg.model.optim.lr == 0.001

    # Assert non-overridden defaults remain
    assert cfg.trainer.accelerator == "auto"
    assert cfg.data.batch_size == 2


def test_save_and_load_yaml():
    """Test YAML I/O utilities."""
    test_data = {"foo": 123, "bar": {"nested": "value"}}

    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_yaml(test_data, tmp_path)
        loaded = read_yaml(tmp_path)

        assert loaded["foo"] == 123
        assert loaded["bar"]["nested"] == "value"
    finally:
        tmp_path.unlink()


def test_defaults_yaml_validates():
    """Test that defaults.yaml is valid and complete."""
    from importlib.resources import files

    defaults_path = files("dtt.config").joinpath("defaults.yaml")
    cfg_dict = read_yaml(defaults_path)
    cfg = Config.model_validate(cfg_dict)

    # Validate key fields
    assert cfg.seed == 42
    assert cfg.trainer.max_epochs == 2
    assert cfg.model.name == "monai.unet"
    assert cfg.data.name == "medical2d"
    assert cfg.logger.wandb.project == "dtt"
    assert cfg.logger.wandb.mode == "offline"
    assert cfg.callbacks.model_checkpoint.monitor == "val/loss"
    assert cfg.callbacks.early_stopping.patience == 5


def test_config_schema_optional_fields():
    """Test that minimal configs work (all defaults apply)."""
    minimal = {"seed": 1, "trainer": {"max_epochs": 1}}
    cfg = Config.model_validate(minimal)

    # Defaults should fill in
    assert cfg.model.name == "monai.unet"
    assert cfg.data.name == "medical2d"
    assert cfg.trainer.accelerator == "auto"
