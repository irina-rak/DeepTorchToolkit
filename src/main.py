import typer
from trogon import Trogon
from typer.main import get_group

import src.commands.app_train as train

app = typer.Typer(
    name="pybiscus_paroma_app",
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)
app.add_typer(train.app, name="train")


@app.command()
def tui(ctx: typer.Context):
    Trogon(get_group(app), click_context=ctx).run()


@app.callback()
def explain():
    """

    DeepTorchToolkit is a CLI tool to easily build a deep learning pipeline with PyTorch Lightning.

    * `train` to launch a training session.

    ---

    To get more information about a specific command, type `deeptoolkit [command] --help`.
    This tool is built on top of PyTorch Lightning, and uses Typer for the CLI part and Rich for the output.
    DeepTorchToolkit strongly relies on Pydantic for configuration validation to ensure a smooth experience.
    """


if __name__ == "__main__":
    app()
