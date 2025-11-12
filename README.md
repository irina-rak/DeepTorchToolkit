# DeepTorchToolkit (DTT)

A clean, intuitive, and extensible template for deep learning projects using:
- PyTorch Lightning (training orchestration)
- MONAI (medical imaging building blocks)
- Weights & Biases (experiment tracking)

This template is optimized for clarity and maintainability, with minimal magic and clear extension points.

## Highlights
- Simple, readable project structure under `src/dtt`
- YAML configs validated with Pydantic models (typo-proof with schemas)
- Typer-based CLI: `dtt train config.yaml` (simple positional argument)
- W&B integration with safe defaults (offline supported)
- Example MONAI U-Net Lightning module and a 2D medical DataModule
- **Registry pattern** with auto-discovery for models and datamodules
- **LR Scheduler support** (cosine, reduce_on_plateau, step, exponential)
- **MONAI Metrics integration** (DiceMetric with extensible pattern)
- Dev tooling: ruff, black, pytest, pre-commit
- CI: fast, torch-free smoke tests by default
- **21 comprehensive tests** covering registry, config, CLI, schedulers, metrics

## Quickstart

1) Create and activate a Python 3.10+ environment.

2) Install DTT for development (light deps only):

```bash
uv sync --extra dev
```

3) Optional: Install heavy deps when ready to train with GPUs/CPUs:

```bash
# Choose torch build for your platform (CPU/CUDA) as needed
uv sync --extra torch

# MONAI + nibabel (depends on torch)
uv sync --extra monai

# Alternatively, install all at once:
uv sync --extra all
```

4) Check the CLI:

```bash
dtt --help
# Can also be run with: python -m dtt.cli.main --help
```

5) Dry-run: print a resolved config (no training):

```bash
dtt train examples/minimal_config.yaml --print-config
```

6) Train with a config (requires torch + optionally monai):

```bash
dtt train examples/minimal_config.yaml
```

Or use defaults if no config provided:

```bash
dtt train
```

Set `WANDB_MODE=offline` to avoid uploading if you just want local logs.

7) Verify the setup (optional):

```bash
python verify_setup.py
```

## Project structure (short)
```
src/dtt/
  cli/            # Typer CLI
  config/         # Pydantic schemas
  data/           # DataModules and transforms
  models/         # LightningModules (e.g., MONAI UNet)
  training/       # Trainer wiring and callbacks
  utils/          # Logging, seeding, IO, registries
```

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Input                                  │
│  CLI: dtt train my_config.yaml                                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Config Layer                                    │
│  1. Load defaults.yaml (OmegaConf)                                  │
│  2. Merge user YAML overrides                                       │
│  3. Validate with Pydantic schemas                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Registry Resolution                             │
│  • get_model(config.model.name) → builder function                  │
│  • get_datamodule(config.data.name) → builder function              │
└────────────┬────────────────────────────┬───────────────────────────┘
             │                            │
             ▼                            ▼
    ┌────────────────┐          ┌──────────────────┐
    │ Model Builder  │          │ DataModule       │
    │ (monai.<model>)│          │ Builder          │
    │                │          │ (medical2d)      │
    │ • UNet model   │          │                  │
    │ • Loss func    │          │ • train_loader   │
    │ • Optimizer    │          │ • val_loader     │
    │ • Scheduler    │          │ • transforms     │
    │ • Metrics      │          │                  │
    └────────┬───────┘          └────────┬─────────┘
             │                           │
             └──────────┬────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Lightning Trainer                                 │
│  • Callbacks (ModelCheckpoint, EarlyStopping, LRMonitor)            │
│  • Logger (WandB offline/online)                                    │
│  • trainer.fit(model, datamodule)                                   │
└─────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Outputs                                        │
│  • Checkpoints: checkpoints/*.ckpt                                  │
│  • Logs: wandb/ (local or synced)                                   │
│  • Metrics: val/loss, val/dice, train/loss                          │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
- **Explicit over implicit**: Registry pattern with manual decorators, no magic auto-discovery
- **Lazy imports**: Heavy deps (torch/MONAI) imported inside functions → fast CLI, torch-free testing
- **Config validation**: Pydantic catches typos early (e.g., `monitr` → error)
- **Composability**: Swap models/datasets by changing 1 line in YAML

## Extending

### Add a New Model (3 steps)

**1. Create the model file**
```python
# src/dtt/models/my_custom_model.py
from dtt.utils.registry import register_model

@register_model("custom.resnet")
def build_custom_resnet(cfg):
    from lightning.pytorch import LightningModule
    # ... implement your model
    return MyResNetLightning(cfg)
```

**2. Register via import**
```python
# src/dtt/models/__init__.py
from dtt.models.monai import unet
from dtt.models import my_custom_model  # Add this line
```

**3. Use in config**
```yaml
model:
  name: custom.resnet
  params:
    num_classes: 10
```

### Add a New DataModule

Same pattern as models:
1. Create `src/dtt/data/datamodules/my_dataset.py`
2. Decorate builder with `@register_datamodule("my_dataset")`
3. Import in `src/dtt/data/datamodules/__init__.py`
4. Reference in config: `data.name: my_dataset`

### Add a New Scheduler

Already supported! Just configure in YAML:
```yaml
model:
  scheduler:
    name: cosine
    params:
      T_max: 50
      eta_min: 1e-6
```

Supported schedulers: `cosine`, `reduce_on_plateau`, `step`, `exponential`

### Add a New Metric

1. Add metric name to config:
```yaml
model:
  metrics:
    - dice
    - hausdorff  # Or anything if you implement it :)
```

2. Implement in your model's `validation_step` and `on_validation_epoch_end`

---

## Configuration Reference

### Trainer Config
```yaml
trainer:
  max_epochs: 10           # Number of training epochs
  accelerator: auto        # "auto", "cpu", "gpu", "tpu"
  devices: auto            # auto, 1, [0,1], etc.
  precision: 32-true       # "32-true", "16-mixed", "bf16-mixed"
  log_every_n_steps: 50    # Logging frequency
```

### Model Config
```yaml
model:
  name: monai.unet         # Registered model name
  optim:
    name: adam             # adam, adamw, sgd
    lr: 1e-3
    weight_decay: 0.0
  scheduler:
    name: cosine           # null (disabled), cosine, reduce_on_plateau, step, exponential
    params:
      T_max: 50            # Scheduler-specific params
  metrics:
    - dice                 # List of metric names
  params:                  # Model-specific hyperparams
    in_channels: 1
    out_channels: 1
```

### Data Config
```yaml
data:
  name: medical2d          # Registered datamodule name
  batch_size: 4
  num_workers: 4
  params:                  # DataModule-specific params
    json_train: datasets/train_split.json  # Path to training data JSON
    json_val: datasets/val_split.json      # Path to validation data JSON
    cache_rate: 0.5        # Fraction of data to cache in memory (0.0-1.0)
    synthetic: true        # Use synthetic data for testing
```

**JSON Dataset Format:**
The framework uses JSON files for dataset metadata. Each JSON file should contain a list of dictionaries with `image` and `label` keys:

```json
[
  {
    "name": "case_001",
    "image": "path/to/image_001.png",
    "label": "path/to/label_001.png"
  },
  {
    "name": "case_002",
    "image": "path/to/image_002.png",
    "label": "path/to/label_002.png"
  }
]
```

- `name`: (optional) Case identifier for tracking
- `image`: Path to input image (relative or absolute)
- `label`: Path to label/mask image

### Callbacks Config
```yaml
callbacks:
  model_checkpoint:
    monitor: val/loss      # Metric to monitor
    mode: min              # "min" or "max"
    save_top_k: 1          # Number of best models to keep
    save_last: true        # Save last checkpoint
    filename: "epoch{epoch:02d}-valloss{val/loss:.3f}"
  early_stopping:
    monitor: val/loss
    patience: 10           # Epochs to wait before stopping
    mode: min
    min_delta: 0.0         # Minimum change to qualify as improvement
  lr_monitor:
    logging_interval: epoch  # "step" or "epoch"
```

### Logger Config
```yaml
logger:
  wandb:
    project: my_project    # W&B project name
    name: run_name         # Specific run name (optional)
    entity: my_team        # W&B team/user (optional)
    tags: [exp1, ablation] # Tags for organization
    mode: offline          # "offline" or "online"
    api_key: null          # Optional: W&B API key
                           # 3 ways to authenticate:
                           # 1. Set in config (here)
                           # 2. Set WANDB_API_KEY env var
                           # 3. Run `wandb login` once
```

---

## Notes
- Add a new model: create a file in `src/dtt/models/...` and register it in `registry.py` with `@register_model("model.name")`.
- Add a new DataModule: add to `src/dtt/data/datamodules/...` and register it with `@register_datamodule("data.name")`.
- Reference them by name in your config (e.g., `model.name: monai.unet`, `data.name: medical2d`).

## Notes
- Heavy imports (torch/monai) are placed inside functions where possible to keep the template importable without them.
- The included tests are torch-free smoke checks; real training tests are marked `@pytest.mark.heavy`.

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.