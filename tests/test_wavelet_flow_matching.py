"""Simple smoke test for Wavelet Flow Matching implementation."""

import torch
from omegaconf import OmegaConf

# Test 1: Module import
print("Test 1: Module import...")
from dtt.models.wavelet_flow_matching import build_wavelet_flow_matching  # noqa: E402

print("✓ Module imported successfully")

# Test 2: Config loading
print("\nTest 2: Config loading...")
cfg = OmegaConf.load("configs/wavelet_flow_matching/wavelet_flow_matching_2d.yaml")
cfg = OmegaConf.to_container(cfg, resolve=True)
print("✓ Config loaded successfully")
print(f"  - Model name: {cfg['model']['name']}")
print(f"  - Spatial dims: {cfg['model']['params']['spatial_dims']}")
print(f"  - Wavelet: {cfg['model']['params']['wavelet']}")

# Test 3: Model instantiation
print("\nTest 3: Model instantiation...")
cfg["data"]["params"]["synthetic"] = True
cfg["data"]["batch_size"] = 2
cfg["trainer"]["devices"] = [0]

model = build_wavelet_flow_matching(cfg)
print("✓ Model instantiated successfully")
print(f"  - Model type: {type(model).__name__}")
print(f"  - Spatial dims: {model.spatial_dims}")
print(f"  - Wavelet: {model.wavelet}")
print(f"  - Apply wavelet transform: {model.apply_wavelet_transform}")
print(f"  - Base channels: {model.base_channels}")

# Test 4: Forward pass
print("\nTest 4: Forward pass...")
model.eval()
with torch.no_grad():
    # Create dummy input IN WAVELET SPACE (4 subbands for 2D)
    batch_size = 2
    num_subbands = 4  # 4 for 2D wavelet
    H_wavelet, W_wavelet = 64, 64  # Half of original 128x128
    x = torch.randn(batch_size, num_subbands, H_wavelet, W_wavelet)
    t = torch.rand(batch_size)

    # Forward pass
    output = model.forward(x, t)

    print("✓ Forward pass successful")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"

# Test 5: DWT/IDWT round-trip
print("\nTest 5: DWT/IDWT round-trip...")
if model.apply_wavelet_transform:
    import torch.nn.functional as F  # noqa: N812

    # Create test image
    test_img = torch.randn(1, 1, 64, 64)

    # Apply DWT -> IDWT
    subbands = model.dwt(test_img)
    stacked = torch.cat(subbands, dim=1)
    subbands_split = torch.chunk(stacked, 4, dim=1)
    reconstructed = model.idwt(*subbands_split)

    # Check reconstruction error
    error = F.mse_loss(reconstructed, test_img).item()
    print("✓ DWT/IDWT round-trip successful")
    print(f"  - Reconstruction MSE: {error:.2e}")
    assert error < 1e-6, f"Reconstruction error too high: {error}"
else:
    print("  - Skipped (wavelet transform disabled)")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
