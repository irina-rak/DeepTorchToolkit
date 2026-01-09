"""Wavelet Flow Matching models.

This package implements Wavelet Flow Matching (WFM) which combines wavelet
transform compression with flow matching dynamics for memory-efficient
high-resolution medical image generation.

References:
    - Friedrich et al., "WDM: 3D Wavelet Diffusion Models for High-Resolution
      Medical Image Synthesis" (DGM4MICCAI 2024)
    - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
"""

from dtt.models.wavelet_flow_matching.wavelet_flow_matching import build_wavelet_flow_matching

__all__ = ["build_wavelet_flow_matching"]
