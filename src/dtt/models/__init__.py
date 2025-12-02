"""Models package - auto-registers all model builders on import."""

# Import model modules to trigger registration decorators
from dtt.models.flow_matching import flow_matching  # noqa: F401
from dtt.models.latent_diffusion import ldm_unet  # noqa: F401
from dtt.models.latent_diffusion import vae  # noqa: F401
from dtt.models.latent_diffusion import vqvae  # noqa: F401
from dtt.models.unet import unet  # noqa: F401
