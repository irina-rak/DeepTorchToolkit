"""Latent Diffusion Model components.

This package provides modular components for Latent Diffusion Models:
- VAE (AutoencoderKL): KL-regularized variational autoencoder
- VQ-VAE: Vector-quantized variational autoencoder
- LDM U-Net: Diffusion model operating in latent space

Training workflow:
1. Train VAE or VQ-VAE on your dataset
2. Train LDM U-Net using the frozen pre-trained autoencoder
3. Generate samples in latent space, decode to pixel space
"""

from dtt.models.latent_diffusion import (
    ldm_unet,  # noqa: F401
    vae,  # noqa: F401
    vqvae,  # noqa: F401
)
