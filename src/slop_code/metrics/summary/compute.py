"""Main entry point for run summary computation.

This module provides the orchestrating function that combines all aggregators
to produce a complete RunSummary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slop_code.metrics.models import RunSummary
from slop_code.metrics.summary import aggregators
from slop_code.metrics.summary.stats import group_by_problem


def compute_run_summary(
    config: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    expected_checkpoints: int,
    run_dir: Path | None = None,
) -> RunSummary:
    """Compute complete summary statistics from checkpoint data.

    Args:
        config: Run configuration dictionary.
        checkpoints: List of checkpoint data dictionaries.
        expected_checkpoints: Total checkpoints the run was configured
            to attempt (sum across the problem list). Used as the
            pct_checkpoints_* denominator; if the agent crashed, the
            produced count is less than this and missing checkpoints
            count as unsolved.
        run_dir: Run output directory; when provided and a run-level
            sloppiness report exists, a compact sloppiness summary is
            attached (generic dict, no model change per new metric).

    Returns:
        RunSummary with all computed statistics.
    """
    problems = group_by_problem(checkpoints)

    sloppiness_summary = (
        _sloppiness_summary(run_dir, checkpoints) if run_dir else None
    )

    return RunSummary.model_validate(
        {
            "model": config["model"]["name"],
            "thinking": config["thinking"],
            "prompt": Path(config["prompt_path"]).stem,
            "agent_type": config["agent"]["type"],
            "agent_version": config["agent"].get("version"),
            "num_problems": len(problems),
            "num_checkpoints": len(checkpoints),
            "expected_checkpoints": expected_checkpoints,
            "costs": aggregators.compute_costs_stats(checkpoints, problems),
            "time": aggregators.compute_time_stats(checkpoints, problems),
            "tokens": aggregators.compute_tokens_stats(checkpoints, problems),
            "steps": aggregators.compute_steps_stats(checkpoints, problems),
            **aggregators.compute_solve_rates(
                checkpoints, problems, expected_checkpoints
            ),
            "pass_rates": aggregators.compute_pass_rates_stats(
                checkpoints, problems
            ),
            "cc": aggregators.compute_cc_stats(checkpoints),
            "ratios": aggregators.compute_ratios_stats(checkpoints),
            **aggregators.compute_composite_scores(checkpoints),
            "sloppiness": sloppiness_summary,
        }
    )


def _sloppiness_summary(
    run_dir: Path, checkpoints: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Compact sloppiness summary from the run-level report.

    Generic dict (coverage counts + per-problem trend availability) so
    new metrics require no model changes. Returns None when the run was
    never measured.
    """
    from slop_code.sloppiness import run_summary_fields

    summary = run_summary_fields(run_dir)
    if not summary:
        return None

    # Checkpoint coverage from the rows themselves.
    measured = [c for c in checkpoints if c.get("sloppiness.status")]
    summary["problems_with_measurements"] = len(
        {c.get("problem") for c in measured if c.get("problem")}
    )
    summary["checkpoints_with_measurements"] = len(measured)
    return summary
