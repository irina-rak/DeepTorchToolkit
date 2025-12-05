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
    checkpoint: Path | None = typer.Option(None, "--checkpoint", "-ckpt", help="Override checkpoint path from config"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Override output directory"),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b", help="Override batch size"),
    num_batches: int | None = typer.Option(None, "--num-batches", "-n", help="Number of batches to process (for unconditional generation)"),
    no_data: bool = typer.Option(False, "--no-data", help="Run without test data (unconditional generation from noise)"),
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
    
    # Handle unconditional generation (no test data)
    if no_data:
        if "inference" not in cfg_dict:
            cfg_dict["inference"] = {}
        cfg_dict["inference"]["use_test_data"] = False
        
        if num_batches is not None:
            cfg_dict["inference"]["num_batches"] = num_batches
    
    if num_batches is not None and not no_data:
        # For test data mode, limit number of batches
        if "trainer" not in cfg_dict:
            cfg_dict["trainer"] = {}
        cfg_dict["trainer"]["limit_test_batches"] = num_batches
    
    # Validate config
    cfg = Config.model_validate(cfg_dict)
    
    # Set seed for reproducibility
    seed_everything(cfg.seed)
    
    # Run inference
    console.log(f"[bold cyan]Checkpoint:[/bold cyan] {checkpoint_path}")
    run_inference(cfg.model_dump(), str(checkpoint_path))


def main() -> None:
    # Execute as a Click command
    get_command(_typer_app)()


if __name__ == "__main__":
    main()

# Expose a Click-compatible command object for testing and CLI usage
app = get_command(_typer_app)
