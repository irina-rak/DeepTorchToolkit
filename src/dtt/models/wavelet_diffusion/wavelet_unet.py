"""Wavelet Diffusion UNet Model.

This module implements a UNet architecture designed for wavelet-based diffusion.
The model operates on wavelet-transformed images (stacked subbands), enabling
memory-efficient processing of high-resolution 3D medical images (up to 256^3).

Key features from the original WDM paper:
- Frequency-aware skip connections: DWT decomposes features, IDWT reconstructs
- Progressive wavelet input: Wavelet residuals added at each encoder level
- Output ResBlocks: Additional processing after decoder

Based on: Friedrich et al., "WDM: 3D Wavelet Diffusion Models for High-Resolution
Medical Image Synthesis" (DGM4MICCAI 2024)
"""

from __future__ import annotations

import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dtt.models.wavelet_diffusion.dwt_idwt import DWT_2D, DWT_3D, IDWT_2D, IDWT_3D

__all__ = ["WaveletDiffusionUNet", "WaveletDiffusionUNet2D", "WaveletDiffusionUNet3D"]


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: Tensor of shape (N,) with timestep values
        dim: Embedding dimension
        max_period: Maximum period for sinusoidal encoding

    Returns:
        Tensor of shape (N, dim) with timestep embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def normalization(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Create a group normalization layer."""
    return nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Create an N-dimensional convolution layer."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dims: {dims}")


def avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Create an N-dimensional average pooling layer."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported dims: {dims}")


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    """Abstract base class for modules with timestep conditioning."""

    @abstractmethod
    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Apply the module with timestep embedding."""
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential module that passes timestep embeddings to children that need it."""

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """Upsampling layer with optional frequency-aware wavelet reconstruction.

    When use_freq=True:
        - Accepts tuple (features, skip_connections) where skip_connections are
          7 high-frequency subbands from the corresponding Downsample
        - Uses IDWT to reconstruct upsampled features from LLL + detail subbands
        - Applies grouped convolution on skip connections before IDWT

    When use_freq=False:
        - Standard interpolation-based upsampling
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
        resample_2d: bool = True,
        use_freq: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.use_freq = use_freq

        # Note: When use_freq=True, resample_2d is ignored because DWT/IDWT
        # always operates on all spatial dimensions (can't do 2D-only 3D wavelet)
        self.resample_2d = resample_2d if not use_freq else False

        if use_freq:
            # IDWT for wavelet reconstruction
            self.idwt = IDWT_3D("haar") if dims == 3 else IDWT_2D("haar")
            # Grouped convolution on 7 (3D) or 3 (2D) high-frequency subbands
            num_detail_subbands = 7 if dims == 3 else 3
            if use_conv:
                self.conv = conv_nd(
                    dims,
                    self.channels * num_detail_subbands,
                    self.out_channels * num_detail_subbands,
                    3,
                    padding=1,
                    groups=num_detail_subbands,
                )
        elif use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: Tensor, skip: tuple[Tensor, ...] | None = None) -> tuple[Tensor, None]:
        """Forward pass.

        Args:
            x: Features tensor (LLL subband if use_freq=True)
            skip: Tuple of high-frequency subbands from corresponding Downsample
                  (7 for 3D, 3 for 2D). Only used when use_freq=True.

        Returns:
            Tuple of (upsampled_features, None) to maintain consistent interface
        """
        assert x.shape[1] == self.channels

        if self.use_freq and skip is not None:
            # Apply grouped conv on skip connections with scaling
            num_detail = 7 if self.dims == 3 else 3
            if self.use_conv:
                skip_cat = torch.cat(skip, dim=1) / 3.0
                skip_cat = self.conv(skip_cat) * 3.0
                skip = tuple(torch.chunk(skip_cat, num_detail, dim=1))

            # IDWT reconstruction: LLL * 3 + detail subbands
            if self.dims == 3:
                x = self.idwt(
                    3.0 * x, skip[0], skip[1], skip[2], skip[3], skip[4], skip[5], skip[6]
                )
            else:
                x = self.idwt(3.0 * x, skip[0], skip[1], skip[2])
        else:
            # Standard interpolation upsampling
            if self.dims == 3 and self.resample_2d:
                x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            if self.use_conv and not self.use_freq:
                x = self.conv(x)

        return x, None


class Downsample(nn.Module):
    """Downsampling layer with optional frequency-aware wavelet decomposition.

    When use_freq=True:
        - Uses DWT to decompose into LLL + 7 detail subbands (3D) or LL + 3 (2D)
        - Returns tuple (LLL / 3.0, (detail_subbands)) for skip connections

    When use_freq=False:
        - Standard strided convolution or average pooling
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
        resample_2d: bool = True,
        use_freq: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.use_freq = use_freq

        if use_freq:
            # DWT for wavelet decomposition
            self.dwt = DWT_3D("haar") if dims == 3 else DWT_2D("haar")
        else:
            stride = (1, 2, 2) if dims == 3 and resample_2d else 2
            if use_conv:
                self.op = conv_nd(
                    dims, self.channels, self.out_channels, 3, stride=stride, padding=1
                )
            else:
                assert self.channels == self.out_channels
                self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        """Forward pass.

        Returns:
            If use_freq=True: tuple (LLL / 3.0, (detail_subbands))
            If use_freq=False: downsampled tensor
        """
        assert x.shape[1] == self.channels

        if self.use_freq:
            subbands = self.dwt(x)
            lll = subbands[0] / 3.0  # Scale LLL for energy balance
            details = subbands[
                1:
            ]  # LLH, LHL, LHH, HLL, HLH, HHL, HHH (7 for 3D) or LH, HL, HH (3 for 2D)
            return lll, details
        else:
            return self.op(x)


class WaveletDownsample(nn.Module):
    """Progressive wavelet input processor.

    Applies DWT to input and projects stacked subbands to match encoder feature size.
    Used to add wavelet residuals at each encoder level.
    """

    def __init__(self, in_ch: int, out_ch: int, dims: int = 3):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dims = dims

        self.dwt = DWT_3D("haar") if dims == 3 else DWT_2D("haar")
        num_subbands = 8 if dims == 3 else 4
        self.conv = conv_nd(dims, in_ch * num_subbands, out_ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply DWT and project to output channels."""
        subbands = self.dwt(x)
        x = torch.cat(subbands, dim=1) / 3.0  # Scale for energy balance
        return self.conv(x)


class ResBlock(TimestepBlock):
    """Residual block with timestep conditioning.

    Supports optional up/downsampling with frequency-aware wavelet skip connections.

    When use_freq=True:
        - For downsampling: returns (features, wavelet_skip_connections)
        - For upsampling: accepts (features, wavelet_skip_connections) input
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        up: bool = False,
        down: bool = False,
        num_groups: int = 32,
        resample_2d: bool = True,
        use_freq: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_groups = num_groups
        self.up = up
        self.down = down
        self.use_freq = use_freq

        self.in_layers = nn.Sequential(
            normalization(channels, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, resample_2d=resample_2d, use_freq=use_freq)
            self.x_upd = Upsample(channels, False, dims, resample_2d=resample_2d, use_freq=use_freq)
        elif down:
            self.h_upd = Downsample(
                channels, False, dims, resample_2d=resample_2d, use_freq=use_freq
            )
            self.x_upd = Downsample(
                channels, False, dims, resample_2d=resample_2d, use_freq=use_freq
            )
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, self.num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: Tensor | tuple, emb: Tensor) -> Tensor | tuple:
        """Forward pass with optional frequency-aware skip connections.

        Args:
            x: Input tensor, or tuple (features, skip_connections) for upsampling
            emb: Timestep embedding

        Returns:
            Output tensor, or tuple (features, skip_connections) for downsampling
        """
        # Handle tuple input (for frequency-aware upsampling)
        h_skip = None
        if isinstance(x, tuple):
            h_skip = x[1]  # Skip connections from downsample
            x = x[0]

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)

            if self.up and self.use_freq:
                # Upsampling with frequency-aware skip connections
                h, _ = self.h_upd(h, skip=h_skip)
                x, _ = self.x_upd(x, skip=h_skip)
            elif self.down and self.use_freq:
                # Downsampling returns (features, skip_connections)
                h, h_skip = self.h_upd(h)
                x, x_skip = self.x_upd(x)
                # Use h_skip for output (they should be the same)
            else:
                # Standard up/downsampling
                result_h = self.h_upd(h)
                result_x = self.x_upd(x)
                # Handle both tuple and non-tuple returns
                if isinstance(result_h, tuple):
                    h, h_skip = result_h
                    x, _ = result_x
                else:
                    h = result_h
                    x = result_x
            h = in_conv(h)
        else:
            if isinstance(x, tuple):
                x = x[0]  # Extract features if tuple
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        out = self.skip_connection(x) + h

        # When use_freq=True, ALWAYS return tuple (like original wunet.py line 267)
        # This ensures proper skip connection propagation through the network
        if self.use_freq:
            return out, h_skip
        return out


class AttentionBlock(nn.Module):
    """Self-attention block for spatial attention."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            if channels % num_head_channels != 0:
                raise ValueError(
                    f"AttentionBlock: channels ({channels}) must be divisible by "
                    f"num_head_channels ({num_head_channels}). "
                    f"Got channels={channels}, num_head_channels={num_head_channels}"
                )
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels, num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_flat))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """QKV attention module."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: Tensor) -> Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class WaveletDiffusionUNet(nn.Module):
    """UNet for Wavelet Diffusion Models.

    This UNet operates on wavelet-transformed images with optional frequency-aware
    skip connections and progressive wavelet input, as described in the original
    WDM paper.

    Args:
        image_size: Input image size (used for attention resolution calculation)
        in_channels: Number of input channels (typically 1 for grayscale, 8 for wavelet subbands)
        model_channels: Base channel count
        out_channels: Number of output channels (typically same as in_channels)
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Downsample rates at which to apply attention
        dropout: Dropout rate
        channel_mult: Channel multipliers for each resolution level
        dims: Spatial dimensions (2 for 2D, 3 for 3D)
        num_classes: Number of classes for conditional generation (optional)
        num_heads: Number of attention heads
        num_head_channels: Channels per attention head (-1 to use num_heads directly)
        num_groups: Number of groups for group normalization
        resblock_updown: Whether to use residual blocks for up/downsampling
        additive_skips: Whether to use additive skip connections
        bottleneck_attention: Whether to use attention in bottleneck
        use_freq: Whether to use frequency-aware wavelet skip connections
        progressive_input: 'residual' to add wavelet input at each level, None to disable
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: tuple[int, ...] | list[int],
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        dims: int = 2,
        num_classes: int | None = None,
        num_heads: int = 1,
        num_head_channels: int | list[int] = -1,
        num_groups: int = 32,
        resblock_updown: bool = True,
        additive_skips: bool = False,
        bottleneck_attention: bool = True,
        resample_2d: bool = True,
        use_freq: bool = False,
        progressive_input: str | None = None,  # 'residual' or None
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_groups = num_groups
        self.additive_skips = additive_skips
        self.bottleneck_attention = bottleneck_attention
        self.dims = dims
        self.use_freq = use_freq
        self.progressive_input = progressive_input
        self.resample_2d = resample_2d
        self.resblock_updown = resblock_updown

        # Convert num_head_channels to list
        if isinstance(num_head_channels, list):
            if len(num_head_channels) != len(channel_mult):
                raise ValueError(
                    f"Length of num_head_channels ({len(num_head_channels)}) must match "
                    f"length of channel_mult ({len(channel_mult)})"
                )
            self.num_head_channels_list = list(num_head_channels)
        elif num_head_channels == -1:
            self.num_head_channels_list = [-1] * len(channel_mult)
        else:
            self.num_head_channels_list = [num_head_channels * mult for mult in channel_mult]

        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Class embedding (optional)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Progressive input WaveletDownsample blocks (if enabled)
        self.progressive_input_blocks = nn.ModuleList()
        if progressive_input == "residual":
            input_pyramid_ch = in_channels
            for _level, mult in enumerate(channel_mult[:-1]):  # Skip last level
                out_ch = model_channels * mult
                self.progressive_input_blocks.append(
                    WaveletDownsample(in_ch=input_pyramid_ch, out_ch=out_ch, dims=dims)
                )
                input_pyramid_ch = out_ch

        # Input block
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # Encoder
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        num_groups=num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    # Use per-level num_head_channels
                    level_head_channels = self.num_head_channels_list[level]
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=level_head_channels,
                            num_groups=num_groups,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            # Downsample
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            down=True,
                            num_groups=num_groups,
                            resample_2d=resample_2d,
                            use_freq=use_freq,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            True,
                            dims=dims,
                            out_channels=out_ch,
                            resample_2d=resample_2d,
                            use_freq=use_freq,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                num_groups=num_groups,
                resample_2d=resample_2d,
            ),
            *(
                [
                    AttentionBlock(
                        ch,
                        num_heads=num_heads,
                        num_head_channels=self.num_head_channels_list[-1],  # Use last level
                        num_groups=num_groups,
                    )
                ]
                if bottleneck_attention
                else []
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                num_groups=num_groups,
                resample_2d=resample_2d,
            ),
        )

        # Decoder
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                mid_ch = (
                    model_channels * mult
                    if not (additive_skips or use_freq)
                    else (input_block_chans[-1] if input_block_chans else model_channels)
                )
                # When use_freq or additive_skips, we use additive skip connections (no concat)
                # so input channels = ch, not ch + ich
                use_additive = additive_skips or use_freq
                layers = [
                    ResBlock(
                        ch if use_additive else ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=mid_ch,
                        dims=dims,
                        num_groups=num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                if ds in attention_resolutions:
                    # Find the correct num_head_channels based on actual channel count
                    # not the decoder level, since additive_skips can change mid_ch
                    ch_per_level = [model_channels * m for m in channel_mult]
                    try:
                        ch_idx = ch_per_level.index(mid_ch)
                        level_head_channels = self.num_head_channels_list[ch_idx]
                    except ValueError:
                        # Fallback: use the level if exact match not found
                        level_head_channels = self.num_head_channels_list[level]

                    layers.append(
                        AttentionBlock(
                            mid_ch,
                            num_heads=num_heads,
                            num_head_channels=level_head_channels,
                            num_groups=num_groups,
                        )
                    )
                ch = mid_ch

                # Upsample
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            mid_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            up=True,
                            num_groups=num_groups,
                            resample_2d=resample_2d,
                            use_freq=use_freq,
                        )
                        if resblock_updown
                        else Upsample(
                            mid_ch,
                            True,
                            dims=dims,
                            out_channels=out_ch,
                            resample_2d=resample_2d,
                            use_freq=use_freq,
                        )
                    )
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output ResBlocks (after decoder, before final output)
        # Uses same num_res_blocks as encoder/decoder (like original wunet.py)
        self.out_res = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.out_res.append(
                TimestepEmbedSequential(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=ch,
                        dims=dims,
                        num_groups=num_groups,
                        resample_2d=resample_2d,
                    )
                )
            )

        # Output
        # NOTE: Original wunet.py does NOT use zero_module here - it blocks gradient flow!
        self.out = nn.Sequential(
            normalization(ch, num_groups),
            nn.SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )

        self.input_block_chans_bk = input_block_chans[:]

    def forward(self, x: Tensor, timesteps: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C, *spatial_dims)
            timesteps: Timestep values of shape (N,)
            y: Optional class labels of shape (N,)

        Returns:
            Output tensor of same shape as x
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # hs stores skip connections:
        # - When use_freq=True: stores wavelet subbands (or None)
        # - When use_freq=False: stores feature maps for concatenation
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y is not None and y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # Progressive input pyramid
        input_pyramid = x
        progressive_idx = 0

        h = x
        for module in self.input_blocks:
            h = module(h, emb)

            # When use_freq=True, extract and store wavelet subbands (skip) only
            # When use_freq=False, store feature maps for concatenation
            skip = None
            if isinstance(h, tuple):
                h, skip = h

            if self.use_freq:
                hs.append(skip)  # Store wavelet skip (or None)
            else:
                hs.append(h)  # Store feature map

            # Apply progressive wavelet input after downsample
            if (
                self.progressive_input == "residual"
                and not self.resample_2d
                and progressive_idx < len(self.progressive_input_blocks)
                and skip is not None
            ):  # skip exists = downsample happened with use_freq
                input_pyramid = self.progressive_input_blocks[progressive_idx](input_pyramid)
                h = h + input_pyramid
                progressive_idx += 1

        h = self.middle_block(h, emb)
        if isinstance(h, tuple):
            h = h[0]

        # Track current skip for frequency-aware upsampling
        current_skip = None

        for module in self.output_blocks:
            new_hs = hs.pop()

            # Update current skip if available
            if new_hs is not None and self.use_freq:
                current_skip = new_hs

            # Use additive skip connections
            if self.additive_skips:
                h = (h + new_hs) / 2**0.5
            # Use frequency-aware skip connections (like original wunet.py lines 777-783)
            elif self.use_freq:
                # Pass wavelet skip to upsample layer via tuple
                if isinstance(h, tuple):
                    # Replace None with stored skip
                    h = (h[0], current_skip)
                else:
                    h = (h, current_skip)
            # Concatenation skip connections (default)
            else:
                h = torch.cat([h, new_hs], dim=1)

            h = module(h, emb)

        # Output ResBlocks
        for module in self.out_res:
            h = module(h, emb)

        # Extract from tuple if needed (like original wunet.py line 794: h, _ = h)
        if isinstance(h, tuple):
            h = h[0]

        return self.out(h)


class WaveletDiffusionUNet2D(WaveletDiffusionUNet):
    """2D Wavelet Diffusion UNet convenience class."""

    def __init__(self, **kwargs):
        kwargs["dims"] = 2
        super().__init__(**kwargs)


class WaveletDiffusionUNet3D(WaveletDiffusionUNet):
    """3D Wavelet Diffusion UNet convenience class."""

    def __init__(self, **kwargs):
        kwargs["dims"] = 3
        super().__init__(**kwargs)
