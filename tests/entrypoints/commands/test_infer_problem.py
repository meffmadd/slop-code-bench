"""Thinking preset validation at the inference CLI boundary."""

from __future__ import annotations

import typing as tp
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from slop_code.common.llms import ThinkingPreset
from slop_code.entrypoints.commands import infer_problem

if tp.TYPE_CHECKING:
    from click.testing import Result


def _invoke_infer_problem(tmp_path: Path, *thinking_args: str) -> Result:
    app = typer.Typer()
    infer_problem.register(app, "infer-problem")
    prompt = tmp_path / "prompt.jinja"
    prompt.write_text("Solve the problem.")
    return CliRunner().invoke(
        app,
        [
            "test-problem",
            "--agent",
            "unused-agent.yaml",
            "--environment-config-path",
            "unused-environment.yaml",
            "--output-path",
            str(tmp_path / "outputs"),
            "--prompt-template",
            str(prompt),
            "--model",
            "aqueduct/nonexistent-test-model",
            *thinking_args,
        ],
    )


@pytest.mark.parametrize("preset", tp.get_args(ThinkingPreset))
def test_infer_problem_accepts_shared_thinking_presets(
    tmp_path: Path, preset: ThinkingPreset
) -> None:
    result = _invoke_infer_problem(tmp_path, "--thinking", preset)

    # Stop at the missing model, before credential resolution or execution.
    assert result.exit_code == 1
    assert "Unknown model 'nonexistent-test-model'" in result.output
    assert "Invalid --thinking" not in result.output


def test_infer_problem_rejects_unknown_thinking_preset(
    tmp_path: Path,
) -> None:
    result = _invoke_infer_problem(tmp_path, "--thinking", "maximum")

    assert result.exit_code == 1
    assert "Invalid --thinking value 'maximum'" in result.output
    assert "Unknown model" not in result.output


def test_infer_problem_rejects_preset_with_token_budget(
    tmp_path: Path,
) -> None:
    result = _invoke_infer_problem(
        tmp_path, "--thinking", "max", "--max-thinking-tokens", "1000"
    )

    assert result.exit_code == 1
    assert "Cannot specify both --thinking and --max-thinking-tokens" in (
        result.output
    )
