"""Models package - auto-registers all model builders on import."""

# Import model modules to trigger registration decorators
from dtt.models.flow_matching import flow_matching  # noqa: F401
from dtt.models.latent_diffusion import (
    ldm_unet,  # noqa: F401
    vae,  # noqa: F401
    vqvae,  # noqa: F401
)
from dtt.models.unet import unet  # noqa: F401
from dtt.models.wavelet_diffusion import wavelet_diffusion  # noqa: F401
from dtt.models.wavelet_flow_matching import wavelet_flow_matching  # noqa: F401

