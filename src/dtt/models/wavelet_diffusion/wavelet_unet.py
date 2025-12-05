"""Wavelet Diffusion UNet Model.

This module implements a UNet architecture designed for wavelet-based diffusion.
The key innovation is that wavelet transforms (DWT/IDWT) are integrated directly
into the network for down/upsampling, with learnable gating mechanisms.

This approach replaces the need for a VAE/VQVAE for latent compression, making
it memory-efficient for high-resolution 3D medical images (up to 256^3).

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
    """Standard upsampling with optional convolution."""

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
        resample_2d: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3 and self.resample_2d:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Standard downsampling with optional convolution."""

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
        resample_2d: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = (1, 2, 2) if dims == 3 and resample_2d else 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class WaveletGatingDownsample2D(nn.Module):
    """Wavelet-gated downsampling for 2D images.

    Uses 2D DWT to decompose input into 4 subbands (LL, LH, HL, HH),
    applies learned gating to weight each subband, and sums them.
    This results in a 2x downsampling in each spatial dimension.
    """

    def __init__(self, channels: int, temb_dim: int, wavelet: str = "haar"):
        super().__init__()
        self.dwt = DWT_2D(wavelet)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fnn = nn.Sequential(
            nn.Linear(channels + temb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 4),  # 4 subbands for 2D
        )
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        # Compute gating values
        p = self.pooling(x).squeeze(-1).squeeze(-1)
        c = torch.cat([p, temb], dim=1)
        gating_values = self.act(self.fnn(c))

        # Apply DWT decomposition
        wavelet_subbands = self.dwt(x)

        # Weight and sum subbands
        scaled_subbands = [
            band * gating.unsqueeze(-1).unsqueeze(-1)
            for band, gating in zip(wavelet_subbands, torch.split(gating_values, 1, dim=1))
        ]
        return sum(scaled_subbands)


class WaveletGatingUpsample2D(nn.Module):
    """Wavelet-gated upsampling for 2D images.

    Expands channels by 4x, applies learned gating, and uses IDWT
    to reconstruct the upsampled image.
    """

    def __init__(self, channels: int, temb_dim: int, wavelet: str = "haar"):
        super().__init__()
        self.idwt = IDWT_2D(wavelet)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fnn = nn.Sequential(
            nn.Linear(channels + temb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 4),  # 4 subbands for 2D
        )
        self.act = nn.Sigmoid()
        self.conv_exp = nn.Conv2d(channels, channels * 4, kernel_size=1)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        # Compute gating values
        p = self.pooling(x).squeeze(-1).squeeze(-1)
        c = torch.cat([p, temb], dim=1)
        gating_values = self.act(self.fnn(c))

        # Expand channels and split into subbands
        wavelet_subbands = self.conv_exp(x).chunk(4, dim=1)

        # Weight subbands
        scaled_subbands = [
            band * gating.unsqueeze(-1).unsqueeze(-1)
            for band, gating in zip(wavelet_subbands, torch.split(gating_values, 1, dim=1))
        ]

        # Apply IDWT reconstruction
        return self.idwt(*scaled_subbands[:4])


class WaveletGatingDownsample3D(nn.Module):
    """Wavelet-gated downsampling for 3D volumes.

    Uses 3D DWT to decompose input into 8 subbands,
    applies learned gating to weight each subband, and sums them.
    This results in a 2x downsampling in each spatial dimension.
    """

    def __init__(self, channels: int, temb_dim: int, wavelet: str = "haar"):
        super().__init__()
        self.dwt = DWT_3D(wavelet)
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.fnn = nn.Sequential(
            nn.Linear(channels + temb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 8),  # 8 subbands for 3D
        )
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        # Compute gating values
        p = self.pooling(x).squeeze(-1).squeeze(-1).squeeze(-1)
        c = torch.cat([p, temb], dim=1)
        gating_values = self.act(self.fnn(c))

        # Apply DWT decomposition
        wavelet_subbands = self.dwt(x)

        # Weight and sum subbands
        scaled_subbands = [
            band * gating.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            for band, gating in zip(wavelet_subbands, torch.split(gating_values, 1, dim=1))
        ]
        return sum(scaled_subbands)


class WaveletGatingUpsample3D(nn.Module):
    """Wavelet-gated upsampling for 3D volumes.

    Expands channels by 8x, applies learned gating, and uses IDWT
    to reconstruct the upsampled volume.
    """

    def __init__(self, channels: int, temb_dim: int, wavelet: str = "haar"):
        super().__init__()
        self.idwt = IDWT_3D(wavelet)
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.fnn = nn.Sequential(
            nn.Linear(channels + temb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 8),  # 8 subbands for 3D
        )
        self.act = nn.Sigmoid()
        self.conv_exp = nn.Conv3d(channels, channels * 8, kernel_size=1)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        # Compute gating values
        p = self.pooling(x).squeeze(-1).squeeze(-1).squeeze(-1)
        c = torch.cat([p, temb], dim=1)
        gating_values = self.act(self.fnn(c))

        # Expand channels and split into subbands
        wavelet_subbands = self.conv_exp(x).chunk(8, dim=1)

        # Weight subbands
        scaled_subbands = [
            band * gating.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            for band, gating in zip(wavelet_subbands, torch.split(gating_values, 1, dim=1))
        ]

        # Apply IDWT reconstruction
        return self.idwt(*scaled_subbands[:8])


class ResBlock(TimestepBlock):
    """Residual block with timestep conditioning.

    Supports optional up/downsampling with wavelet-gated or standard methods.
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
        use_wavelet_updown: bool = False,
        wavelet: str = "haar",
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_groups = num_groups

        self.in_layers = nn.Sequential(
            normalization(channels, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        self.use_wavelet = use_wavelet_updown and self.updown

        if up:
            if use_wavelet_updown:
                if dims == 2:
                    self.h_upd = WaveletGatingUpsample2D(channels, emb_channels, wavelet)
                    self.x_upd = WaveletGatingUpsample2D(channels, emb_channels, wavelet)
                else:
                    self.h_upd = WaveletGatingUpsample3D(channels, emb_channels, wavelet)
                    self.x_upd = WaveletGatingUpsample3D(channels, emb_channels, wavelet)
            else:
                self.h_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
                self.x_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
        elif down:
            if use_wavelet_updown:
                if dims == 2:
                    self.h_upd = WaveletGatingDownsample2D(channels, emb_channels, wavelet)
                    self.x_upd = WaveletGatingDownsample2D(channels, emb_channels, wavelet)
                else:
                    self.h_upd = WaveletGatingDownsample3D(channels, emb_channels, wavelet)
                    self.x_upd = WaveletGatingDownsample3D(channels, emb_channels, wavelet)
            else:
                self.h_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
                self.x_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
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

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            if self.use_wavelet:
                h = self.h_upd(h, emb)
                x = self.x_upd(x, emb)
            else:
                h = self.h_upd(h)
                x = self.x_upd(x)
            h = in_conv(h)
        else:
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

        return self.skip_connection(x) + h


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
            assert channels % num_head_channels == 0
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

    This UNet operates on wavelet-transformed images and uses wavelet-gated
    up/downsampling for memory efficiency. The diffusion process happens
    in wavelet space, allowing training on high-resolution images.

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
        num_head_channels: Channels per attention head (-1 to use num_heads)
        num_groups: Number of groups for group normalization
        use_wavelet_updown: Whether to use wavelet-gated up/downsampling
        wavelet: Wavelet type ('haar', 'db2', etc.)
        resblock_updown: Whether to use residual blocks for up/downsampling
        additive_skips: Whether to use additive skip connections
        bottleneck_attention: Whether to use attention in bottleneck
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
        num_head_channels: int = -1,
        num_groups: int = 32,
        use_wavelet_updown: bool = True,
        wavelet: str = "haar",
        resblock_updown: bool = True,
        additive_skips: bool = False,
        bottleneck_attention: bool = True,
        resample_2d: bool = True,
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
        self.use_wavelet_updown = use_wavelet_updown
        self.wavelet = wavelet
        self.additive_skips = additive_skips
        self.bottleneck_attention = bottleneck_attention
        self.dims = dims

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
                        use_wavelet_updown=False,
                        wavelet=wavelet,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
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
                            use_wavelet_updown=use_wavelet_updown,
                            wavelet=wavelet,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, True, dims=dims, out_channels=out_ch, resample_2d=resample_2d
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
                        num_head_channels=num_head_channels,
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
                    if not additive_skips
                    else (input_block_chans[-1] if input_block_chans else model_channels)
                )
                layers = [
                    ResBlock(
                        ch + ich if not additive_skips else ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mid_ch,
                        dims=dims,
                        num_groups=num_groups,
                        resample_2d=resample_2d,
                        use_wavelet_updown=False,
                        wavelet=wavelet,
                    )
                ]
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            mid_ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
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
                            use_wavelet_updown=use_wavelet_updown,
                            wavelet=wavelet,
                        )
                        if resblock_updown
                        else Upsample(
                            mid_ch, True, dims=dims, out_channels=out_ch, resample_2d=resample_2d
                        )
                    )
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output
        self.out = nn.Sequential(
            normalization(ch, num_groups),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
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

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y is not None and y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            new_hs = hs.pop()
            if self.additive_skips:
                h = (h + new_hs) / 2
            else:
                h = torch.cat([h, new_hs], dim=1)
            h = module(h, emb)

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
