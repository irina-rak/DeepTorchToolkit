"""DataModules package - auto-registers all datamodule builders on import."""

# Import datamodule modules to trigger registration decorators
from dtt.data.datamodules import base, chest_xray2d, medical2d  # noqa: F401
