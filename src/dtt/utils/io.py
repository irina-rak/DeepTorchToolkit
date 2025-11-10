from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def read_yaml(path: str | Path) -> Dict[str, Any]:
    """Read a YAML file into a plain dict using OmegaConf (resolves interpolations)."""
    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conf = OmegaConf.create(obj)
    OmegaConf.save(conf, str(path))
