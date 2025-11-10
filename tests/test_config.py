from __future__ import annotations

from importlib.resources import files

from dtt.config.schemas import Config
from dtt.utils.io import read_yaml


def test_load_defaults_config():
    defaults_path = files("dtt.config").joinpath("defaults.yaml")
    cfg_dict = read_yaml(defaults_path)
    cfg = Config.model_validate(cfg_dict)
    assert cfg.trainer.max_epochs >= 1
