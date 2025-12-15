"""PyTorch layers for 1D, 2D, and 3D Discrete Wavelet Transforms (DWT) and Inverse DWT (IDWT).

This module implements DWT and IDWT as nn.Module layers, making them easy to integrate
into neural networks. The transforms use matrix multiplication for efficiency and
support automatic differentiation.

Supported wavelet families (via pywt):
    - Haar: 'haar'
    - Daubechies: 'db1', 'db2', ..., 'db38'
    - Biorthogonal: 'bior1.1', 'bior1.3', ..., 'bior6.8'
    - Reverse Biorthogonal: 'rbio1.1', ..., 'rbio6.8'
    - Coiflets: 'coif1', ..., 'coif17'
    - Symlets: 'sym2', ..., 'sym20'

Note:
    Exact reconstruction is guaranteed only when input dimensions are even and
    the wavelet filter length is 2. For other cases, boundary effects may cause
    small reconstruction errors.

Based on: https://github.com/pfriedri/wdm-3d (MIT License)
Original: https://github.com/LiQiufu/WaveCNet (CC BY-NC-SA 4.0)
"""

from __future__ import annotations

import numpy as np
import pywt
import torch
from torch import Tensor, nn

from .dwt_idwt_functions import (
    DWTFunction_1D,
    DWTFunction_2D,
    DWTFunction_2D_tiny,
    DWTFunction_3D,
    IDWTFunction_1D,
    IDWTFunction_2D,
    IDWTFunction_3D,
)

__all__ = [
    "DWT_1D",
    "IDWT_1D",
    "DWT_2D",
    "DWT_2D_tiny",
    "IDWT_2D",
    "DWT_3D",
    "IDWT_3D",
]


class DWT_1D(nn.Module):
    """1D Discrete Wavelet Transform layer.

    Decomposes 1D signals into low-frequency (approximation) and
    high-frequency (detail) components.

    Args:
        wavename: Name of the wavelet (e.g., 'haar', 'db2', 'bior1.3')

    Input shape: (N, C, Length)
    Output: tuple of (lfc, hfc) each with shape (N, C, Length/2)
    """

    def __init__(self, wavename: str):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = self.band_length // 2

        # These will be set dynamically based on input size
        self.matrix_low: Tensor | None = None
        self.matrix_high: Tensor | None = None
        self.input_height: int = 0

    def _get_matrix(self, device: torch.device):
        """Generate transformation matrices for the current input size."""
        L1 = self.input_height
        L = L1 // 2
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2

        matrix_h = matrix_h[:, (self.band_length_half - 1) : end]
        matrix_g = matrix_g[:, (self.band_length_half - 1) : end]

        self.matrix_low = torch.tensor(matrix_h, dtype=torch.float32, device=device)
        self.matrix_high = torch.tensor(matrix_g, dtype=torch.float32, device=device)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        """Apply 1D DWT decomposition.

        Args:
            input: Input tensor of shape (N, C, Length)

        Returns:
            Tuple of (low_frequency, high_frequency) components
        """
        assert len(input.size()) == 3
        self.input_height = input.size(-1)
        self._get_matrix(input.device)
        return DWTFunction_1D.apply(input, self.matrix_low, self.matrix_high)


class IDWT_1D(nn.Module):
    """1D Inverse Discrete Wavelet Transform layer.

    Reconstructs the original signal from low and high frequency components.

    Args:
        wavename: Name of the wavelet (must match the one used for decomposition)

    Input: tuple of (lfc, hfc) each with shape (N, C, Length/2)
    Output shape: (N, C, Length)
    """

    def __init__(self, wavename: str):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = list(wavelet.rec_lo)
        self.band_high = list(wavelet.rec_hi)
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = self.band_length // 2

        self.matrix_low: Tensor | None = None
        self.matrix_high: Tensor | None = None
        self.input_height: int = 0

    def _get_matrix(self, device: torch.device):
        """Generate transformation matrices for the current input size."""
        L1 = self.input_height
        L = L1 // 2
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2

        matrix_h = matrix_h[:, (self.band_length_half - 1) : end]
        matrix_g = matrix_g[:, (self.band_length_half - 1) : end]

        self.matrix_low = torch.tensor(matrix_h, dtype=torch.float32, device=device)
        self.matrix_high = torch.tensor(matrix_g, dtype=torch.float32, device=device)

    def forward(self, L: Tensor, H: Tensor) -> Tensor:
        """Reconstruct signal from wavelet components.

        Args:
            L: Low-frequency component
            H: High-frequency component

        Returns:
            Reconstructed signal
        """
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size(-1) + H.size(-1)
        self._get_matrix(L.device)
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)


class DWT_2D_tiny(nn.Module):
    """2D DWT that only returns the low-frequency component.

    This is a memory-efficient version used when only the approximation
    coefficient is needed (e.g., for downsampling in networks).

    Args:
        wavename: Name of the wavelet

    Input shape: (N, C, H, W)
    Output shape: (N, C, H/2, W/2)
    """

    def __init__(self, wavename: str):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = self.band_length // 2

        self.input_height: int = 0
        self.input_width: int = 0
        self.matrix_low_0: Tensor | None = None
        self.matrix_low_1: Tensor | None = None
        self.matrix_high_0: Tensor | None = None
        self.matrix_high_1: Tensor | None = None

    def _get_matrix(self, device: torch.device):
        """Generate transformation matrices."""
        L1 = max(self.input_height, self.input_width)
        L = L1 // 2
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[
            : (self.input_height // 2), : (self.input_height + self.band_length - 2)
        ]
        matrix_h_1 = matrix_h[
            : (self.input_width // 2), : (self.input_width + self.band_length - 2)
        ]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[
            : (self.input_height - self.input_height // 2),
            : (self.input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            : (self.input_width - self.input_width // 2),
            : (self.input_width + self.band_length - 2),
        ]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)

        self.matrix_low_0 = torch.tensor(matrix_h_0, dtype=torch.float32, device=device)
        self.matrix_low_1 = torch.tensor(matrix_h_1, dtype=torch.float32, device=device)
        self.matrix_high_0 = torch.tensor(matrix_g_0, dtype=torch.float32, device=device)
        self.matrix_high_1 = torch.tensor(matrix_g_1, dtype=torch.float32, device=device)

    def forward(self, input: Tensor) -> Tensor:
        """Apply 2D DWT and return only low-frequency component.

        Args:
            input: Input tensor of shape (N, C, H, W)

        Returns:
            Low-frequency component of shape (N, C, H/2, W/2)
        """
        assert len(input.size()) == 4
        self.input_height = input.size(-2)
        self.input_width = input.size(-1)
        self._get_matrix(input.device)
        return DWTFunction_2D_tiny.apply(
            input,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_high_0,
            self.matrix_high_1,
        )


class DWT_2D(nn.Module):
    """2D Discrete Wavelet Transform layer.

    Decomposes 2D images into four subbands:
    - LL: Low-Low (approximation)
    - LH: Low-High (horizontal details)
    - HL: High-Low (vertical details)
    - HH: High-High (diagonal details)

    Args:
        wavename: Name of the wavelet (e.g., 'haar', 'db2', 'bior1.3')

    Input shape: (N, C, H, W)
    Output: tuple of (LL, LH, HL, HH) each with shape (N, C, H/2, W/2)
    """

    def __init__(self, wavename: str):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = self.band_length // 2

        self.input_height: int = 0
        self.input_width: int = 0
        self.matrix_low_0: Tensor | None = None
        self.matrix_low_1: Tensor | None = None
        self.matrix_high_0: Tensor | None = None
        self.matrix_high_1: Tensor | None = None

    def _get_matrix(self, device: torch.device):
        """Generate transformation matrices."""
        L1 = max(self.input_height, self.input_width)
        L = L1 // 2
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[
            : (self.input_height // 2), : (self.input_height + self.band_length - 2)
        ]
        matrix_h_1 = matrix_h[
            : (self.input_width // 2), : (self.input_width + self.band_length - 2)
        ]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[
            : (self.input_height - self.input_height // 2),
            : (self.input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            : (self.input_width - self.input_width // 2),
            : (self.input_width + self.band_length - 2),
        ]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)

        self.matrix_low_0 = torch.tensor(matrix_h_0, dtype=torch.float32, device=device)
        self.matrix_low_1 = torch.tensor(matrix_h_1, dtype=torch.float32, device=device)
        self.matrix_high_0 = torch.tensor(matrix_g_0, dtype=torch.float32, device=device)
        self.matrix_high_1 = torch.tensor(matrix_g_1, dtype=torch.float32, device=device)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply 2D DWT decomposition.

        Args:
            input: Input tensor of shape (N, C, H, W)

        Returns:
            Tuple of (LL, LH, HL, HH) subbands
        """
        assert len(input.size()) == 4
        self.input_height = input.size(-2)
        self.input_width = input.size(-1)
        self._get_matrix(input.device)
        return DWTFunction_2D.apply(
            input,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_high_0,
            self.matrix_high_1,
        )


class IDWT_2D(nn.Module):
    """2D Inverse Discrete Wavelet Transform layer.

    Reconstructs the original 2D image from four subbands.

    Args:
        wavename: Name of the wavelet (must match the one used for decomposition)

    Input: tuple of (LL, LH, HL, HH) each with shape (N, C, H/2, W/2)
    Output shape: (N, C, H, W)
    """

    def __init__(self, wavename: str):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = list(wavelet.rec_lo)
        self.band_high = list(wavelet.rec_hi)
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = self.band_length // 2

        self.input_height: int = 0
        self.input_width: int = 0
        self.matrix_low_0: Tensor | None = None
        self.matrix_low_1: Tensor | None = None
        self.matrix_high_0: Tensor | None = None
        self.matrix_high_1: Tensor | None = None

    def _get_matrix(self, device: torch.device):
        """Generate transformation matrices."""
        L1 = max(self.input_height, self.input_width)
        L = L1 // 2
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[
            : (self.input_height // 2), : (self.input_height + self.band_length - 2)
        ]
        matrix_h_1 = matrix_h[
            : (self.input_width // 2), : (self.input_width + self.band_length - 2)
        ]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[
            : (self.input_height - self.input_height // 2),
            : (self.input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            : (self.input_width - self.input_width // 2),
            : (self.input_width + self.band_length - 2),
        ]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)

        self.matrix_low_0 = torch.tensor(matrix_h_0, dtype=torch.float32, device=device)
        self.matrix_low_1 = torch.tensor(matrix_h_1, dtype=torch.float32, device=device)
        self.matrix_high_0 = torch.tensor(matrix_g_0, dtype=torch.float32, device=device)
        self.matrix_high_1 = torch.tensor(matrix_g_1, dtype=torch.float32, device=device)

    def forward(self, LL: Tensor, LH: Tensor, HL: Tensor, HH: Tensor) -> Tensor:
        """Reconstruct image from wavelet subbands.

        Args:
            LL: Low-Low subband (approximation)
            LH: Low-High subband (horizontal details)
            HL: High-Low subband (vertical details)
            HH: High-High subband (diagonal details)

        Returns:
            Reconstructed image
        """
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size(-2) + HH.size(-2)
        self.input_width = LL.size(-1) + HH.size(-1)
        self._get_matrix(LL.device)
        return IDWTFunction_2D.apply(
            LL,
            LH,
            HL,
            HH,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_high_0,
            self.matrix_high_1,
        )


class DWT_3D(nn.Module):
    """3D Discrete Wavelet Transform layer.

    Decomposes 3D volumes into eight subbands:
    - LLL: Low-Low-Low (approximation)
    - LLH, LHL, LHH, HLL, HLH, HHL, HHH: Various detail subbands

    The naming convention is (Depth, Height, Width) where L=Low, H=High.

    Args:
        wavename: Name of the wavelet (e.g., 'haar', 'db2')

    Input shape: (N, C, D, H, W)
    Output: tuple of 8 tensors, each with shape (N, C, D/2, H/2, W/2)
    """

    def __init__(self, wavename: str):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = self.band_length // 2

        self.input_depth: int = 0
        self.input_height: int = 0
        self.input_width: int = 0
        self.matrix_low_0: Tensor | None = None
        self.matrix_low_1: Tensor | None = None
        self.matrix_low_2: Tensor | None = None
        self.matrix_high_0: Tensor | None = None
        self.matrix_high_1: Tensor | None = None
        self.matrix_high_2: Tensor | None = None

    def _get_matrix(self, device: torch.device):
        """Generate transformation matrices for all three dimensions."""
        L1 = max(self.input_height, self.input_width)
        L = L1 // 2
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[
            : (self.input_height // 2), : (self.input_height + self.band_length - 2)
        ]
        matrix_h_1 = matrix_h[
            : (self.input_width // 2), : (self.input_width + self.band_length - 2)
        ]
        matrix_h_2 = matrix_h[
            : (self.input_depth // 2), : (self.input_depth + self.band_length - 2)
        ]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[
            : (self.input_height - self.input_height // 2),
            : (self.input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            : (self.input_width - self.input_width // 2),
            : (self.input_width + self.band_length - 2),
        ]
        matrix_g_2 = matrix_g[
            : (self.input_depth - self.input_depth // 2),
            : (self.input_depth + self.band_length - 2),
        ]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1) : end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1) : end]

        self.matrix_low_0 = torch.tensor(matrix_h_0, dtype=torch.float32, device=device)
        self.matrix_low_1 = torch.tensor(matrix_h_1, dtype=torch.float32, device=device)
        self.matrix_low_2 = torch.tensor(matrix_h_2, dtype=torch.float32, device=device)
        self.matrix_high_0 = torch.tensor(matrix_g_0, dtype=torch.float32, device=device)
        self.matrix_high_1 = torch.tensor(matrix_g_1, dtype=torch.float32, device=device)
        self.matrix_high_2 = torch.tensor(matrix_g_2, dtype=torch.float32, device=device)

    def forward(
        self, input: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply 3D DWT decomposition.

        Args:
            input: Input tensor of shape (N, C, D, H, W)

        Returns:
            Tuple of 8 subbands: (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
        """
        assert len(input.size()) == 5
        self.input_depth = input.size(-3)
        self.input_height = input.size(-2)
        self.input_width = input.size(-1)
        self._get_matrix(input.device)
        return DWTFunction_3D.apply(
            input,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_low_2,
            self.matrix_high_0,
            self.matrix_high_1,
            self.matrix_high_2,
        )


class IDWT_3D(nn.Module):
    """3D Inverse Discrete Wavelet Transform layer.

    Reconstructs the original 3D volume from eight subbands.

    Args:
        wavename: Name of the wavelet (must match the one used for decomposition)

    Input: tuple of 8 tensors, each with shape (N, C, D/2, H/2, W/2)
    Output shape: (N, C, D, H, W)
    """

    def __init__(self, wavename: str):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = list(wavelet.rec_lo)
        self.band_high = list(wavelet.rec_hi)
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = self.band_length // 2

        self.input_depth: int = 0
        self.input_height: int = 0
        self.input_width: int = 0
        self.matrix_low_0: Tensor | None = None
        self.matrix_low_1: Tensor | None = None
        self.matrix_low_2: Tensor | None = None
        self.matrix_high_0: Tensor | None = None
        self.matrix_high_1: Tensor | None = None
        self.matrix_high_2: Tensor | None = None

    def _get_matrix(self, device: torch.device):
        """Generate transformation matrices for all three dimensions."""
        L1 = max(self.input_height, self.input_width)
        L = L1 // 2
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[
            : (self.input_height // 2), : (self.input_height + self.band_length - 2)
        ]
        matrix_h_1 = matrix_h[
            : (self.input_width // 2), : (self.input_width + self.band_length - 2)
        ]
        matrix_h_2 = matrix_h[
            : (self.input_depth // 2), : (self.input_depth + self.band_length - 2)
        ]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[
            : (self.input_height - self.input_height // 2),
            : (self.input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            : (self.input_width - self.input_width // 2),
            : (self.input_width + self.band_length - 2),
        ]
        matrix_g_2 = matrix_g[
            : (self.input_depth - self.input_depth // 2),
            : (self.input_depth + self.band_length - 2),
        ]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1) : end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1) : end]

        self.matrix_low_0 = torch.tensor(matrix_h_0, dtype=torch.float32, device=device)
        self.matrix_low_1 = torch.tensor(matrix_h_1, dtype=torch.float32, device=device)
        self.matrix_low_2 = torch.tensor(matrix_h_2, dtype=torch.float32, device=device)
        self.matrix_high_0 = torch.tensor(matrix_g_0, dtype=torch.float32, device=device)
        self.matrix_high_1 = torch.tensor(matrix_g_1, dtype=torch.float32, device=device)
        self.matrix_high_2 = torch.tensor(matrix_g_2, dtype=torch.float32, device=device)

    def forward(
        self,
        LLL: Tensor,
        LLH: Tensor,
        LHL: Tensor,
        LHH: Tensor,
        HLL: Tensor,
        HLH: Tensor,
        HHL: Tensor,
        HHH: Tensor,
    ) -> Tensor:
        """Reconstruct volume from wavelet subbands.

        Args:
            LLL: Low-Low-Low subband (approximation)
            LLH, LHL, LHH, HLL, HLH, HHL, HHH: Detail subbands

        Returns:
            Reconstructed 3D volume
        """
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == 5
        assert len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        self.input_depth = LLL.size(-3) + HHH.size(-3)
        self.input_height = LLL.size(-2) + HHH.size(-2)
        self.input_width = LLL.size(-1) + HHH.size(-1)
        self._get_matrix(LLL.device)
        return IDWTFunction_3D.apply(
            LLL,
            LLH,
            LHL,
            LHH,
            HLL,
            HLH,
            HHL,
            HHH,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_low_2,
            self.matrix_high_0,
            self.matrix_high_1,
            self.matrix_high_2,
        )
