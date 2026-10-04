"""Tests for the pi/DeepSeek experiment configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from slop_code.agent_runner.agents.pi import PiAgent
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.registry import build_agent_config
from slop_code.common.llms import ModelCatalog
from slop_code.common.llms import ThinkingPreset
from slop_code.entrypoints.config.loader import load_run_config

RUN_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs/runs/pi-deepseek.yaml"
)


@pytest.mark.parametrize(
    ("override", "expected_level"),
    [(None, "max"), ("low", "low"), ("none", None), ("disabled", "off")],
)
def test_pi_deepseek_thinking_config_and_overrides(
    override: ThinkingPreset | None,
    expected_level: str | None,
) -> None:
    overrides = [] if override is None else [f"thinking={override}"]
    run_config = load_run_config(
        config_path=RUN_CONFIG_PATH,
        cli_overrides=overrides,
    )

    assert run_config.model.provider == "aqueduct"
    assert run_config.model.name == "deepseek-v4-vision-exp"
    assert run_config.thinking == (override or "max")
    assert f"_just-solve_{run_config.thinking}_" in run_config.output_path
    assert run_config.model_dump(mode="json")["thinking"] == (
        override or "max"
    )

    model = ModelCatalog.get(run_config.model.name)
    assert model is not None
    credential = ProviderCredential(
        provider="aqueduct",
        credential_type=CredentialType.ENV_VAR,
        value="fake-key",
        source="AQUEDUCT_API_KEY",
        destination_key="AQUEDUCT_API_KEY",
    )
    agent = PiAgent._from_config(
        config=build_agent_config(run_config.agent),
        model=model,
        credential=credential,
        problem_name="test-problem",
        verbose=False,
        image="test-image",
        thinking_preset=run_config.thinking,
        thinking_max_tokens=run_config.thinking_max_tokens,
    )
    assert isinstance(agent, PiAgent)
    assert agent.thinking == expected_level

    command = agent._build_command("test prompt")
    if expected_level is None:
        assert "--thinking" not in command
    else:
        thinking_index = command.index("--thinking")
        assert command[thinking_index + 1] == expected_level
