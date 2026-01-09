from __future__ import annotations

from pathlib import Path

import typer
from typer.main import get_command

from dtt.config.schemas import Config
from dtt.inference.infer import run_inference
from dtt.training.train import run_training
from dtt.utils.io import read_yaml
from dtt.utils.logging import get_console
from dtt.utils.seed import seed_everything

_typer_app = typer.Typer(add_completion=False, no_args_is_help=True, help="DeepTorchToolkit CLI")


@_typer_app.command()
def train(
    config: Path | None = typer.Argument(
        None, help="Path to YAML config file (uses defaults if omitted)"
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print resolved config and exit"
    ),
):
    """Train a model specified by the config."""
    console = get_console()
    # Load defaults then merge user config if provided
    from importlib.resources import files

    defaults_path = files("dtt.config").joinpath("defaults.yaml")
    cfg_dict = read_yaml(defaults_path)

    if config is not None:
        user_cfg = read_yaml(config)
        # shallow merge: user overrides default keys
        cfg_dict.update(user_cfg)

    cfg = Config.model_validate(cfg_dict)

    if print_config:

        console.print_json(data=cfg.model_dump())
        raise typer.Exit(code=0)

    seed_everything(cfg.seed)
    run_training(cfg.model_dump())


@_typer_app.command()
def infer(
    config: Path = typer.Argument(..., help="Path to inference config YAML file"),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", "-ckpt", help="Override checkpoint path from config"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override output directory"
    ),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b", help="Override batch size"),
    num_batches: int | None = typer.Option(
        None,
        "--num-batches",
        "-n",
        help="Number of batches to process (for unconditional generation)",
    ),
    no_data: bool = typer.Option(
        False, "--no-data", help="Run without test data (unconditional generation from noise)"
    ),
):
    """Run inference using a trained model checkpoint.

    This command runs inference by calling the model's test_step method through
    the Lightning Trainer.test() API. The test_step implementation handles
    the inference logic, metrics computation, and output saving.

    The checkpoint path is specified in the config file under 'checkpoint_path',
    but can be overridden with the --checkpoint flag.

    Examples:

        # Infer with config (checkpoint path in config)
        dtt infer configs/inference.yaml

        # Override checkpoint path
        dtt infer configs/inference.yaml --checkpoint path/to/checkpoint.ckpt

        # Unconditional generation without test data
        dtt infer configs/inference.yaml --no-data -n 10 -b 16

        # Override output directory
        dtt infer configs/inference.yaml -o ./results
    """
    console = get_console()

    # Load config
    if not config.exists():
        console.log(f"[bold red]Error:[/bold red] Config file not found: {config}")
        raise typer.Exit(code=1)

    cfg_dict = read_yaml(config)

    # Get checkpoint path: from flag override or config file
    if checkpoint is not None:
        checkpoint_path = checkpoint
        console.log(f"[cyan]Using checkpoint from CLI:[/cyan] {checkpoint_path}")
    elif "checkpoint_path" in cfg_dict:
        checkpoint_path = Path(cfg_dict["checkpoint_path"])
        console.log(f"[cyan]Using checkpoint from config:[/cyan] {checkpoint_path}")
    else:
        console.log("[bold red]Error:[/bold red] No checkpoint specified!")
        console.log("Please specify 'checkpoint_path' in config or use --checkpoint flag")
        raise typer.Exit(code=1)

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        console.log(f"[bold red]Error:[/bold red] Checkpoint not found: {checkpoint_path}")
        raise typer.Exit(code=1)

    # Apply CLI overrides
    if output_dir is not None:
        cfg_dict["output_dir"] = str(output_dir)

    if batch_size is not None:
        if "data" not in cfg_dict:
            cfg_dict["data"] = {}
        cfg_dict["data"]["batch_size"] = batch_size

    # Handle unconditional generation
    # --no-data CLI flag overrides config, but config's use_test_data is respected by default
    if no_data:
        if "inference" not in cfg_dict:
            cfg_dict["inference"] = {}
        cfg_dict["inference"]["use_test_data"] = False

    # Check if unconditional mode (either from CLI or config)
    is_unconditional = cfg_dict.get("inference", {}).get("use_test_data", True) is False

    if num_batches is not None:
        if is_unconditional:
            # For unconditional mode, set num_batches in inference config
            if "inference" not in cfg_dict:
                cfg_dict["inference"] = {}
            cfg_dict["inference"]["num_batches"] = num_batches
        else:
            # For test data mode, limit number of batches via trainer
            if "trainer" not in cfg_dict:
                cfg_dict["trainer"] = {}
            cfg_dict["trainer"]["limit_test_batches"] = num_batches

    # Validate config (for type checking)
    cfg = Config.model_validate(cfg_dict)

    # Set seed for reproducibility
    seed_everything(cfg.seed)

    # Run inference - pass original dict to preserve fields like output_dir
    console.log(f"[bold cyan]Checkpoint:[/bold cyan] {checkpoint_path}")
    run_inference(cfg_dict, str(checkpoint_path))


@_typer_app.command()
def evaluate(
    config: Path | None = typer.Argument(
        None, help="Path to evaluation config YAML file (optional)"
    ),
    real_dir: Path | None = typer.Option(
        None, "--real-dir", "-r", help="Directory containing real images"
    ),
    fake_dir: Path | None = typer.Option(
        None, "--fake-dir", "-f", help="Directory containing generated/fake images"
    ),
    spatial_dims: int | None = typer.Option(
        None, "--spatial-dims", "-d", help="Number of spatial dimensions (2 or 3)"
    ),
    feature_extractor: str | None = typer.Option(
        None,
        "--feature-extractor",
        "-e",
        help="Feature extractor type: auto, inception, medicalnet",
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-m", help="Maximum number of samples to load"
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", "-b", help="Batch size for feature extraction"
    ),
    device: str | None = typer.Option(None, "--device", help="Device for computation"),
    no_kid: bool = typer.Option(False, "--no-kid", help="Skip KID computation (FID only)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Path to save results as JSON"),
):
    """Evaluate generated images using FID and KID metrics.

    This command computes distribution-based metrics (FID and KID) to assess
    how similar the generated images are to real images.

    You can either provide a config file OR use CLI arguments directly.
    CLI arguments override config file values when both are provided.

    Metrics:
      - FID (Fréchet Inception Distance): Lower is better
      - KID (Kernel Inception Distance): Lower is better, more robust for small samples

    Examples:

        # Using a config file
        dtt evaluate configs/evaluation/eval_2d.yaml

        # Using CLI arguments
        dtt evaluate -r /path/to/real -f /path/to/generated

        # Config file with CLI overrides
        dtt evaluate configs/eval.yaml --max-samples 1000

        # Evaluate 3D volumes
        dtt evaluate -r /path/to/real -f /path/to/generated -d 3
    """
    from dtt.config.schemas import EvaluationConfig
    from dtt.evaluation import evaluate_generated_images

    console = get_console()

    # Load config from file if provided
    eval_cfg = None
    if config is not None:
        if not config.exists():
            console.log(f"[bold red]Error:[/bold red] Config file not found: {config}")
            raise typer.Exit(code=1)

        cfg_dict = read_yaml(config)
        # Check if evaluation config is nested or at root level
        if "evaluation" in cfg_dict:
            eval_cfg = EvaluationConfig.model_validate(cfg_dict["evaluation"])
        else:
            eval_cfg = EvaluationConfig.model_validate(cfg_dict)

        console.log(f"[cyan]Loaded config from:[/cyan] {config}")

    # Build final config from file + CLI overrides
    final_real_dir = str(real_dir) if real_dir else (eval_cfg.real_dir if eval_cfg else None)
    final_fake_dir = str(fake_dir) if fake_dir else (eval_cfg.fake_dir if eval_cfg else None)
    final_spatial_dims = (
        spatial_dims if spatial_dims else (eval_cfg.spatial_dims if eval_cfg else 2)
    )
    final_feature_extractor = (
        feature_extractor
        if feature_extractor
        else (eval_cfg.feature_extractor if eval_cfg else "auto")
    )
    final_max_samples = max_samples if max_samples else (eval_cfg.max_samples if eval_cfg else None)
    final_batch_size = batch_size if batch_size else (eval_cfg.batch_size if eval_cfg else 32)
    final_device = device if device else (eval_cfg.device if eval_cfg else "cuda")
    final_compute_kid = not no_kid if no_kid else (eval_cfg.compute_kid if eval_cfg else True)
    final_output = str(output) if output else (eval_cfg.output_path if eval_cfg else None)

    # Validate required fields
    if not final_real_dir:
        console.log(
            "[bold red]Error:[/bold red] real_dir is required (use --real-dir or config file)"
        )
        raise typer.Exit(code=1)

    if not final_fake_dir:
        console.log(
            "[bold red]Error:[/bold red] fake_dir is required (use --fake-dir or config file)"
        )
        raise typer.Exit(code=1)

    # Validate directories exist
    if not Path(final_real_dir).exists():
        console.log(f"[bold red]Error:[/bold red] Real directory not found: {final_real_dir}")
        raise typer.Exit(code=1)

    if not Path(final_fake_dir).exists():
        console.log(f"[bold red]Error:[/bold red] Fake directory not found: {final_fake_dir}")
        raise typer.Exit(code=1)

    if final_spatial_dims not in [2, 3]:
        console.log(
            f"[bold red]Error:[/bold red] spatial_dims must be 2 or 3, got {final_spatial_dims}"
        )
        raise typer.Exit(code=1)

    if final_feature_extractor not in ["auto", "inception", "medicalnet"]:
        console.log(
            f"[bold red]Error:[/bold red] Invalid feature_extractor: {final_feature_extractor}"
        )
        raise typer.Exit(code=1)

    # Run evaluation
    # Get target_size from config if available
    final_target_size = eval_cfg.target_size if eval_cfg and eval_cfg.target_size else None

    try:
        evaluate_generated_images(
            real_dir=final_real_dir,
            fake_dir=final_fake_dir,
            spatial_dims=final_spatial_dims,
            feature_extractor=final_feature_extractor,
            max_samples=final_max_samples,
            batch_size=final_batch_size,
            device=final_device,
            compute_kid=final_compute_kid,
            output_path=final_output,
            target_size=final_target_size,
        )
    except Exception as e:
        console.log(f"[bold red]Error during evaluation:[/bold red] {e}")
        raise typer.Exit(code=1) from e


def main() -> None:
    # Execute as a Click command
    get_command(_typer_app)()


if __name__ == "__main__":
    main()

# Expose a Click-compatible command object for testing and CLI usage
app = get_command(_typer_app)
