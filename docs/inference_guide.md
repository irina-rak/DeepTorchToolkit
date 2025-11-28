# Inference Guide for DeepTorchToolkit

This guide explains how to use DTT's inference system to run inference on trained models.

## Overview

DTT's inference system leverages the Lightning module's `test_step` method, making it model-agnostic and consistent with the training pipeline. The inference script:

1. Loads a trained model from a checkpoint
2. Sets up the data pipeline OR runs without data (unconditional generation)
3. Calls `Trainer.test()` which invokes `test_step()` on each batch
4. Saves outputs and optionally computes metrics

**Supported modes:**
- **With test data**: For computing metrics, reconstruction, or conditional generation
- **Without test data** (`--no-data`): For unconditional generation from noise

## Quick Start

### Basic Usage

```bash
# Run inference with test data (for metrics)
dtt infer configs/inference_config.yaml

# Unconditional generation (no test data needed)
dtt infer configs/inference_unconditional.yaml --no-data -n 10

# Override checkpoint path
dtt infer configs/inference_config.yaml --checkpoint path/to/different.ckpt
```

### CLI Options

```bash
dtt infer <config> [OPTIONS]

Arguments:
  config                     Path to inference config YAML file (required)

Options:
  --checkpoint, -ckpt PATH   Override checkpoint path from config
  -o, --output-dir PATH      Override output directory
  -b, --batch-size INT       Override batch size
  -n, --num-batches INT      Number of batches to process
  --no-data                  Run without test data (unconditional generation)
```

### Examples

```bash
# Inference with test data (compute metrics)
dtt infer configs/inference_flow_matching_cxr2d.yaml

# Override checkpoint path
dtt infer configs/inference_flow_matching_cxr2d.yaml \
  --checkpoint experiments/flow_matching_cxr/different_checkpoint.ckpt

# Unconditional generation: 100 samples (10 batches × 10 samples)
dtt infer configs/inference_unconditional.yaml \
  --no-data \
  -n 10 \
  -b 10

# Override output directory
dtt infer configs/inference.yaml \
  --output-dir results/my_experiment
```

## Configuration

### Inference Config Structure

An inference config should include:

1. **Output settings**: Where to save results
2. **Model architecture**: Must match training exactly
3. **Data pipeline**: Test data or synthetic generation
4. **Inference-specific parameters**: Solver steps, quality settings

Example minimal config:

```yaml
# Checkpoint (required)
checkpoint_path: path/to/checkpoint.ckpt

# Global settings
seed: 42
output_dir: outputs
project: my_inference
run_name: run_001

# Trainer (inference mode)
trainer:
  accelerator: gpu
  devices: [0]
  logger: false

# Model (must match training!)
model:
  name: flow_matching
  # ... same as training config ...
  params:
    solver_args:
      time_points: 50  # More steps = better quality
      method: midpoint

# Data (only needed if NOT using --no-data)
data:
  name: data.medical2d
  batch_size: 16
  params:
    json_test: path/to/test.json

# Inference settings (optional)
inference:
  use_test_data: true  # Set to false for unconditional generation
  num_batches: 10      # Only for unconditional mode
```

## Output Structure

Inference results are saved in a structured directory:

```
<output_dir>/<project>/<run_name>/
├── generated_samples/
│   ├── sample_batch0000_idx00.pt   # PyTorch tensor (exact)
│   ├── sample_batch0000_idx00.png  # Visualization
│   ├── sample_batch0000_idx01.pt
│   ├── sample_batch0000_idx01.png
│   └── ...
└── [other outputs from test_step]
```

## Model-Specific Implementation

### Flow Matching Models

For flow matching models, `test_step`:
- Generates samples from random noise using the ODE solver
- Computes PSNR/SSIM metrics if real images are provided
- Saves both `.pt` tensors and `.png` visualizations
- Uses EMA weights if available

**Key parameters for inference quality:**

```yaml
solver_args:
  time_points: 50  # Higher = better quality, slower
  method: midpoint # midpoint recommended for balance
```

### Custom Models

To add inference support to your own model:

1. Implement `test_step()` in your Lightning module:

```python
def test_step(self, batch, batch_idx):
    # 1. Generate/process samples
    # 2. Compute metrics
    # 3. Save outputs using self.inference_output_dir
    # 4. Return metrics dict
    return {"metric_name": value}
```

2. Access the output directory:

```python
if hasattr(self, "inference_output_dir"):
    output_dir = self.inference_output_dir
    # Save your outputs
```

## Tips & Best Practices

### Quality vs Speed

For flow matching models:
- **Fast preview**: `time_points: 10-20`
- **Balanced**: `time_points: 50` (recommended)
- **High quality**: `time_points: 100+`

### GPU Memory

If you run out of memory:
- Reduce `batch_size`
- Use `precision: bf16-mixed`
- Use fewer `devices`

### Reproducibility

Always set `seed` for reproducible generation:

```yaml
seed: 42
```

### Inference Modes

**With test data** (compute metrics against real images):
```bash
dtt infer configs/inference.yaml
```

Config:
```yaml
checkpoint_path: path/to/checkpoint.ckpt
data:
  params:
    json_test: path/to/test_set.json
```

**Unconditional generation** (no test data, pure generation from noise):
```bash
dtt infer configs/inference_unconditional.yaml --no-data -n 10 -b 16
```

Config:
```yaml
checkpoint_path: path/to/checkpoint.ckpt
inference:
  use_test_data: false
  num_batches: 10
data:
  batch_size: 16
```

## Troubleshooting

### "Checkpoint not found"
- Verify the checkpoint path is correct
- Use absolute paths or paths relative to current directory

### "Model architecture mismatch"
- Ensure inference config's `model.params.unet_config` exactly matches training
- Check `data_range`, `max_timestep`, and other preprocessing settings

### "Out of memory"
- Reduce `batch_size`
- Use fewer ODE `time_points`
- Use single GPU: `devices: [0]`

### "No config found"
- Specify config explicitly with `--config`
- Or place a `.yaml` file in the checkpoint directory

## Advanced Usage

### Programmatic API

You can also use the inference API directly in Python:

```python
from dtt.inference.generate import run_inference
from dtt.utils.io import read_yaml

# Load config
cfg = read_yaml("inference_config.yaml")

# For unconditional generation, add:
cfg["inference"] = {
    "use_test_data": False,
    "num_batches": 10
}

# Run inference
run_inference(cfg, checkpoint_path="path/to/checkpoint.ckpt")
```

### Custom Inference Logic

For specialized inference needs, you can:

1. Subclass the model and override `test_step`
2. Add custom callbacks that run during `trainer.test()`
3. Implement custom metrics and logging

## See Also

- Example configs: `configs/inference_*.yaml`
- Training guide: `docs/training.md`
- Model implementations: `src/dtt/models/`
