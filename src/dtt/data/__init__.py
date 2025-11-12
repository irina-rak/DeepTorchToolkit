"""Data package - auto-registers all datamodules on import."""

# Import datamodules to trigger registration decorators
from dtt.data import datamodules  # noqa: F401
