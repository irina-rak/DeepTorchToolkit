# Weights & Biases Authentication

DTT supports multiple ways to authenticate with Weights & Biases, giving you flexibility for different workflows.

## Authentication Methods

### Method 1: Config File (Recommended for Quick Experiments)

Add your API key directly in the config YAML:

```yaml
logger:
  wandb:
    project: my_project
    mode: online
    api_key: "your-api-key-here"  # Get from https://wandb.ai/authorize
```

This allows DTT to use the specified API key for a specific run for quick experiments, and multi-account setups.
API key stored in plain text, so you have to be careful not to commit this file to version control!

---

### Method 2: Environment Variable (Recommended for Security)

Set the environment variable before training:

```bash
export WANDB_API_KEY="your-api-key-here"
python -m dtt.cli.main train --config config.yaml
```

Or add to your `~/.bashrc` or `~/.zshrc`:

```bash
export WANDB_API_KEY="your-api-key-here"
```

**Config:**
```yaml
logger:
  wandb:
    project: my_project
    mode: online
    api_key: null  # Will use WANDB_API_KEY env var
```

This method keeps your API key out of config files and version control. However, you need to ensure the
environment variable is set in each terminal session (or added to your shell profile), otherwise DTT
won't find it!

---

### Method 3: Persistent Login (Recommended for Development)

Run once to authenticate permanently:

```bash
wandb login
```

Then your configs don't need the API key at all:

```yaml
logger:
  wandb:
    project: my_project
    mode: online
    api_key: null  # Will use saved credentials
```

This last method stores your credentials securely in `~/.netrc` and DTT will use them automatically.
It allows to avoid setting env vars or putting keys in config files. However, it ties the credentials
to a specific machine, and you may need to re-login if credentials expire.

---

## Priority Order

If multiple authentication methods are present, DTT uses this priority:

1. **Config file** (`api_key` in YAML)
2. **Environment variable** (`WANDB_API_KEY`)
3. **Persistent login** (from `wandb login`)

---

## Example Workflows

### Workflow 1: Development (Persistent Login)

```bash
# One-time setup
wandb login

# Train anytime
python -m dtt.cli.main train --config config.yaml
```

### Workflow 2: Production (Environment Variable)

```bash
# Set API key securely
export WANDB_API_KEY=$(cat ~/.secrets/wandb_key)

# Train
python -m dtt.cli.main train --config config.yaml
```

### Workflow 3: Quick Experiment (Config File)

```yaml
# config_experiment.yaml
logger:
  wandb:
    project: quick_test
    mode: online
    api_key: "abc123..."  # Different account for experiments
```

### Workflow 4: Multi-Account Setup

```bash
# Personal experiments
export WANDB_API_KEY="personal-key"
python -m dtt.cli.main train --config personal_config.yaml

# Work experiments (different key in config)
python -m dtt.cli.main train --config work_config.yaml  # Uses api_key from config
```

---

## Security Best Practices

### It is advised to:
- Use environment variables for production
- Use persistent login for development machines
- Add `*.yaml` with API keys to `.gitignore`
- Store keys in secure vaults (e.g., `~/.secrets/`)

### Avoid:
- Commit API keys to version control (Always double-check!)
- Share configs with hardcoded keys
- Use the same key across all accounts

---

## Offline Mode (No Authentication Needed)

If you don't want to sync to W&B cloud:

```yaml
logger:
  wandb:
    project: my_project
    mode: offline  # Logs locally only
    api_key: null   # Not needed for offline mode
```

Logs will be saved to `wandb/` directory. You can sync later:

```bash
wandb sync wandb/offline-run-xxx
```

---

## Getting Your API Key

1. Go to https://wandb.ai/authorize
2. Copy your API key
3. Use one of the methods above

---

## Troubleshooting

### Error: "wandb: ERROR authentication required"

**Solution:** Set your API key using one of the three methods above.

### Multiple accounts

Use config file method with different keys per project, or switch env var before training.

### Key not working

- Check that key is correct (no extra spaces)
- Try `wandb login` to re-authenticate
- Verify mode is set to `online` (not `offline`)
