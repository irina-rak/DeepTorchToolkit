from __future__ import annotations

from click.testing import CliRunner

from dtt.cli.main import app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Train a model specified by the config" in result.output
