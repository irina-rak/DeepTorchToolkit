from __future__ import annotations

from pathlib import Path

import typer
from typer.main import get_command

from dtt.config.schemas import Config
from dtt.training.train import run_training
from dtt.utils.io import read_yaml
from dtt.utils.logging import get_console
from dtt.utils.seed import seed_everything

_typer_app = typer.Typer(add_completion=False, no_args_is_help=True, help="DeepTorchToolkit CLI")


@_typer_app.command(help="DeepTorchToolkit CLI - Train a model specified by the config.")
def train(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to YAML config; uses defaults if omitted"
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


def main() -> None:
    # Execute as a Click command
    get_command(_typer_app)()


if __name__ == "__main__":
    main()

# Expose a Click-compatible command object for testing and CLI usage
app = get_command(_typer_app)
