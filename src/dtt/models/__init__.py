"""Models package - auto-registers all model builders on import."""

# Import model modules to trigger registration decorators
from dtt.models.monai import unet  # noqa: F401
