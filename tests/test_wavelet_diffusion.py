"""Tests for Wavelet Diffusion models."""

import pytest
import torch

# Mark tests that require heavy dependencies
pytestmark = pytest.mark.heavy


class TestDWTIDWT:
    """Tests for DWT/IDWT transform layers."""

    def test_dwt_2d_shape(self):
        """Test 2D DWT output shapes."""
        from dtt.models.wavelet_diffusion.dwt_idwt import DWT_2D

        dwt = DWT_2D("haar")
        x = torch.randn(2, 3, 64, 64)
        ll, lh, hl, hh = dwt(x)

        assert ll.shape == (2, 3, 32, 32)
        assert lh.shape == (2, 3, 32, 32)
        assert hl.shape == (2, 3, 32, 32)
        assert hh.shape == (2, 3, 32, 32)

    def test_idwt_2d_reconstruction(self):
        """Test 2D IDWT reconstruction."""
        from dtt.models.wavelet_diffusion.dwt_idwt import DWT_2D, IDWT_2D

        dwt = DWT_2D("haar")
        idwt = IDWT_2D("haar")

        x = torch.randn(2, 3, 64, 64)
        subbands = dwt(x)
        x_recon = idwt(*subbands)

        # Haar wavelet should give exact reconstruction
        assert x_recon.shape == x.shape
        assert torch.allclose(x, x_recon, atol=1e-5)

    def test_dwt_3d_shape(self):
        """Test 3D DWT output shapes."""
        from dtt.models.wavelet_diffusion.dwt_idwt import DWT_3D

        dwt = DWT_3D("haar")
        x = torch.randn(2, 1, 32, 32, 32)
        subbands = dwt(x)

        assert len(subbands) == 8
        for subband in subbands:
            assert subband.shape == (2, 1, 16, 16, 16)

    def test_idwt_3d_reconstruction(self):
        """Test 3D IDWT reconstruction."""
        from dtt.models.wavelet_diffusion.dwt_idwt import DWT_3D, IDWT_3D

        dwt = DWT_3D("haar")
        idwt = IDWT_3D("haar")

        x = torch.randn(2, 1, 32, 32, 32)
        subbands = dwt(x)
        x_recon = idwt(*subbands)

        assert x_recon.shape == x.shape
        assert torch.allclose(x, x_recon, atol=1e-5)

    def test_dwt_different_wavelets(self):
        """Test DWT with different wavelet families."""
        from dtt.models.wavelet_diffusion.dwt_idwt import DWT_2D, IDWT_2D

        wavelets = ["haar", "db2", "bior1.3"]
        x = torch.randn(2, 1, 64, 64)

        for wavelet in wavelets:
            dwt = DWT_2D(wavelet)
            idwt = IDWT_2D(wavelet)
            subbands = dwt(x)
            x_recon = idwt(*subbands)
            # Reconstruction error should be small
            assert torch.allclose(x, x_recon, atol=1e-4), f"Failed for wavelet {wavelet}"


class TestWaveletDiffusionUNet:
    """Tests for Wavelet Diffusion UNet."""

    def test_unet_2d_forward(self):
        """Test 2D UNet forward pass."""
        from dtt.models.wavelet_diffusion.wavelet_unet import WaveletDiffusionUNet2D

        model = WaveletDiffusionUNet2D(
            image_size=64,
            in_channels=4,  # 4 subbands for 2D wavelet
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=[8],
            channel_mult=[1, 2],
            num_heads=2,
            use_wavelet_updown=True,
        )

        x = torch.randn(2, 4, 32, 32)  # Wavelet space (half resolution)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)

        assert out.shape == x.shape

    def test_unet_3d_forward(self):
        """Test 3D UNet forward pass."""
        from dtt.models.wavelet_diffusion.wavelet_unet import WaveletDiffusionUNet3D

        model = WaveletDiffusionUNet3D(
            image_size=32,
            in_channels=8,  # 8 subbands for 3D wavelet
            model_channels=16,
            out_channels=8,
            num_res_blocks=1,
            attention_resolutions=[4],
            channel_mult=[1, 2],
            num_heads=2,
            use_wavelet_updown=True,
        )

        x = torch.randn(2, 8, 16, 16, 16)  # Wavelet space
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)

        assert out.shape == x.shape

    def test_unet_class_conditional(self):
        """Test class-conditional UNet."""
        from dtt.models.wavelet_diffusion.wavelet_unet import WaveletDiffusionUNet

        model = WaveletDiffusionUNet(
            image_size=64,
            in_channels=4,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=[8],
            channel_mult=[1, 2],
            dims=2,
            num_classes=10,
        )

        x = torch.randn(2, 4, 32, 32)
        t = torch.randint(0, 1000, (2,))
        y = torch.randint(0, 10, (2,))
        out = model(x, t, y=y)

        assert out.shape == x.shape


class TestWaveletGating:
    """Tests for wavelet-gated up/downsampling."""

    def test_gating_downsample_2d(self):
        """Test 2D wavelet-gated downsampling."""
        from dtt.models.wavelet_diffusion.wavelet_unet import WaveletGatingDownsample2D

        layer = WaveletGatingDownsample2D(channels=16, temb_dim=64)
        x = torch.randn(2, 16, 32, 32)
        temb = torch.randn(2, 64)
        out = layer(x, temb)

        assert out.shape == (2, 16, 16, 16)

    def test_gating_upsample_2d(self):
        """Test 2D wavelet-gated upsampling."""
        from dtt.models.wavelet_diffusion.wavelet_unet import WaveletGatingUpsample2D

        layer = WaveletGatingUpsample2D(channels=16, temb_dim=64)
        x = torch.randn(2, 16, 16, 16)
        temb = torch.randn(2, 64)
        out = layer(x, temb)

        assert out.shape == (2, 16, 32, 32)

    def test_gating_downsample_3d(self):
        """Test 3D wavelet-gated downsampling."""
        from dtt.models.wavelet_diffusion.wavelet_unet import WaveletGatingDownsample3D

        layer = WaveletGatingDownsample3D(channels=16, temb_dim=64)
        x = torch.randn(2, 16, 16, 16, 16)
        temb = torch.randn(2, 64)
        out = layer(x, temb)

        assert out.shape == (2, 16, 8, 8, 8)

    def test_gating_upsample_3d(self):
        """Test 3D wavelet-gated upsampling."""
        from dtt.models.wavelet_diffusion.wavelet_unet import WaveletGatingUpsample3D

        layer = WaveletGatingUpsample3D(channels=16, temb_dim=64)
        x = torch.randn(2, 16, 8, 8, 8)
        temb = torch.randn(2, 64)
        out = layer(x, temb)

        assert out.shape == (2, 16, 16, 16, 16)


class TestWaveletDiffusionModel:
    """Tests for the full Wavelet Diffusion model."""

    def test_model_build_2d(self):
        """Test building 2D Wavelet Diffusion model from config."""
        from dtt.models.wavelet_diffusion import build_wavelet_diffusion

        config = {
            "model": {
                "name": "wavelet_diffusion",
                "optim": {"name": "adam", "lr": 1e-4},
                "scheduler": {"name": None},
                "params": {
                    "spatial_dims": 2,
                    "wavelet": "haar",
                    "apply_wavelet_transform": True,
                    "unet_config": {
                        "image_size": 64,
                        "in_channels": 1,
                        "model_channels": 16,
                        "out_channels": 1,
                        "num_res_blocks": 1,
                        "attention_resolutions": [8],
                        "channel_mult": [1, 2],
                    },
                    "num_train_timesteps": 100,
                },
            },
            "data": {"name": "test", "batch_size": 2},
            "trainer": {"max_epochs": 1},
        }

        model = build_wavelet_diffusion(config)
        assert model is not None

    def test_model_build_3d(self):
        """Test building 3D Wavelet Diffusion model from config."""
        from dtt.models.wavelet_diffusion import build_wavelet_diffusion

        config = {
            "model": {
                "name": "wavelet_diffusion",
                "optim": {"name": "adam", "lr": 1e-4},
                "scheduler": {"name": None},
                "params": {
                    "spatial_dims": 3,
                    "wavelet": "haar",
                    "apply_wavelet_transform": True,
                    "unet_config": {
                        "image_size": 32,
                        "in_channels": 1,
                        "model_channels": 8,
                        "out_channels": 1,
                        "num_res_blocks": 1,
                        "attention_resolutions": [4],
                        "channel_mult": [1, 2],
                    },
                    "num_train_timesteps": 100,
                },
            },
            "data": {"name": "test", "batch_size": 2},
            "trainer": {"max_epochs": 1},
        }

        model = build_wavelet_diffusion(config)
        assert model is not None


class TestDDPMScheduler:
    """Tests for DDPM noise scheduler."""

    def test_add_noise(self):
        """Test adding noise at different timesteps."""
        from dtt.models.wavelet_diffusion.wavelet_diffusion import build_wavelet_diffusion

        # Build model to access scheduler
        config = {
            "model": {
                "name": "wavelet_diffusion",
                "optim": {"name": "adam", "lr": 1e-4},
                "scheduler": {"name": None},
                "params": {
                    "spatial_dims": 2,
                    "wavelet": "haar",
                    "apply_wavelet_transform": False,
                    "unet_config": {
                        "image_size": 32,
                        "in_channels": 1,
                        "model_channels": 8,
                        "out_channels": 1,
                        "num_res_blocks": 1,
                        "attention_resolutions": [],
                        "channel_mult": [1],
                    },
                    "num_train_timesteps": 100,
                },
            },
            "data": {"name": "test", "batch_size": 2},
            "trainer": {"max_epochs": 1},
        }

        model = build_wavelet_diffusion(config)

        x = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(x)
        t = torch.tensor([0, 99])

        noisy = model.scheduler.add_noise(x, noise, t)

        # At t=0, noisy should be close to x
        # At t=99, noisy should be close to noise
        assert noisy.shape == x.shape
