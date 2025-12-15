"""Wavelet Diffusion models for high-resolution medical image synthesis.

This module implements Wavelet Diffusion Models (WDM) which use wavelet transforms
instead of VAE for image compression in diffusion models. This approach is memory-efficient
and allows training on high-resolution 3D images (256^3) on a single GPU.

References:
    - Friedrich et al., "WDM: 3D Wavelet Diffusion Models for High-Resolution Medical Image Synthesis"
      MICCAI Workshop on Deep Generative Models, 2024
"""

from dtt.models.wavelet_diffusion.wavelet_diffusion import build_wavelet_diffusion

__all__ = ["build_wavelet_diffusion"]
