from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from click.testing import CliRunner

from dtt.cli.main import app


def test_cli_help():
    """Test that CLI help works and shows expected text."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "DeepTorchToolkit CLI" in result.output


def test_cli_train_help():
    """Test that train subcommand help works."""
    import re

    runner = CliRunner()
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    # Remove ANSI color codes for reliable testing
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "config" in clean_output.lower()
    assert "print-config" in clean_output.lower() or "print_config" in clean_output.lower()


def test_cli_print_config():
    """Test that --print-config works and outputs JSON."""
    runner = CliRunner()
    result = runner.invoke(app, ["--print-config"])
    assert result.exit_code == 0

    # Should output valid JSON containing expected keys
    import json

    try:
        output_data = json.loads(result.output)
        assert "seed" in output_data
        assert "trainer" in output_data
        assert "model" in output_data
        assert "data" in output_data
    except json.JSONDecodeError:
        pytest.fail(f"Output is not valid JSON: {result.output}")


def test_cli_print_config_with_custom_yaml():
    """Test that --print-config merges custom YAML."""
    custom_yaml = """
seed: 777
trainer:
  max_epochs: 5
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(custom_yaml)
        tmp_path = Path(tmp.name)

    try:
        runner = CliRunner()
        result = runner.invoke(app, ["--config", str(tmp_path), "--print-config"])
        assert result.exit_code == 0

        import json

        output_data = json.loads(result.output)
        assert output_data["seed"] == 777
        assert output_data["trainer"]["max_epochs"] == 5
    finally:
        tmp_path.unlink()
