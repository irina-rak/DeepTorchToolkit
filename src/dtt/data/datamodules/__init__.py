"""DataModules package - auto-registers all datamodule builders on import."""

# Import datamodule modules to trigger registration decorators
from dtt.data.datamodules import base, medical2d, medical3dct  # noqa: F401
