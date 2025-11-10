from __future__ import annotations

import pytest

from dtt.utils.registry import get_datamodule, get_model, register_datamodule, register_model


def test_register_and_get_model():
    """Test that we can register and retrieve a model builder."""

    @register_model("test_dummy_model")
    def dummy_model_builder(cfg):
        return {"model": "dummy", "config": cfg}

    builder = get_model("test_dummy_model")
    result = builder({"foo": "bar"})
    assert result["model"] == "dummy"
    assert result["config"]["foo"] == "bar"


def test_register_and_get_datamodule():
    """Test that we can register and retrieve a datamodule builder."""

    @register_datamodule("test_dummy_dm")
    def dummy_dm_builder(cfg):
        return {"datamodule": "dummy", "config": cfg}

    builder = get_datamodule("test_dummy_dm")
    result = builder({"baz": "qux"})
    assert result["datamodule"] == "dummy"
    assert result["config"]["baz"] == "qux"


def test_get_nonexistent_model_raises():
    """Test that getting a non-existent model raises KeyError with helpful message."""
    with pytest.raises(KeyError, match="Model 'nonexistent' not registered"):
        get_model("nonexistent")


def test_get_nonexistent_datamodule_raises():
    """Test that getting a non-existent datamodule raises KeyError with helpful message."""
    with pytest.raises(KeyError, match="DataModule 'nonexistent' not registered"):
        get_datamodule("nonexistent")


def test_auto_registered_models_available():
    """Test that models from dtt.models package are auto-registered."""
    # Import the models package to trigger auto-registration
    import dtt.models  # noqa: F401

    # MONAI UNet should be registered
    builder = get_model("monai.unet")
    assert builder is not None
    assert callable(builder)


def test_auto_registered_datamodules_available():
    """Test that datamodules from dtt.data.datamodules package are auto-registered."""
    # Import the datamodules package to trigger auto-registration
    import dtt.data.datamodules  # noqa: F401

    # Base and medical2d should be registered
    builder_base = get_datamodule("base")
    builder_med = get_datamodule("medical2d")
    assert builder_base is not None
    assert builder_med is not None
    assert callable(builder_base)
    assert callable(builder_med)
