from __future__ import annotations

from collections.abc import Callable
from typing import Any

ModelBuilder = Callable[[dict], Any]
DataModuleBuilder = Callable[[dict], Any]

_MODELS: dict[str, ModelBuilder] = {}
_DATAMODULES: dict[str, DataModuleBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(fn: ModelBuilder) -> ModelBuilder:
        _MODELS[name] = fn
        return fn

    return decorator


def get_model(name: str) -> ModelBuilder:
    if name not in _MODELS:
        raise KeyError(f"Model '{name}' not registered. Available: {list(_MODELS)}")
    return _MODELS[name]


def register_datamodule(name: str) -> Callable[[DataModuleBuilder], DataModuleBuilder]:
    def decorator(fn: DataModuleBuilder) -> DataModuleBuilder:
        _DATAMODULES[name] = fn
        return fn

    return decorator


def get_datamodule(name: str) -> DataModuleBuilder:
    if name not in _DATAMODULES:
        raise KeyError(f"DataModule '{name}' not registered. Available: {list(_DATAMODULES)}")
    return _DATAMODULES[name]
