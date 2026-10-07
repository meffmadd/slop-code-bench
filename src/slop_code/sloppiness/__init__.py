"""Sloppiness metrics adapter for SlopCodeBench.

Opt-in integration with the standalone ``sloppiness`` analysis package
(sibling checkout ``sloppiness-metrics``). The adapter is the only
module that touches the standalone package; SCB commands call these
helpers. When the option is disabled, nothing here runs and previously
persisted reports are never erased.

The standalone package is imported lazily: SCB works without it unless
sloppiness measurement is requested. Install it with
``uv sync --group sloppiness``.

Artifacts (defaults):

- Per checkpoint: ``<checkpoint>/quality_analysis/sloppiness.json``
- Per run: ``<run>/sloppiness.json``

All failures are isolated: an analysis error is logged and reported as
a diagnostic; it never changes correctness results, solve rates, agent
stopping policy, or whether other checkpoints run.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from slop_code.common import QUALITY_DIR
from slop_code.logging import get_logger

logger = get_logger(__name__)

SLOPPINESS_FILENAME = "sloppiness.json"
RUN_SLOPPINESS_FILENAME = "sloppiness.json"


class SloppinessUnavailableError(RuntimeError):
    """Raised when measurement is requested but the package is absent."""


def is_available() -> bool:
    """Whether the standalone sloppiness package is importable."""
    try:
        import sloppiness  # noqa: F401
    except ImportError:
        return False
    return True


def _require_package() -> Any:
    try:
        import sloppiness
    except ImportError as e:
        raise SloppinessUnavailableError(
            "The --sloppiness option requires the standalone sloppiness "
            "package. Install it with: uv sync --group sloppiness "
            "(or `uv pip install -e ../sloppiness-metrics`)."
        ) from e
    return sloppiness


def sidecar_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / QUALITY_DIR / SLOPPINESS_FILENAME


def run_report_path(run_dir: Path) -> Path:
    return run_dir / RUN_SLOPPINESS_FILENAME


def measure_run(
    run_dir: Path,
    problems: list[str] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Measure every available snapshot in a run and persist reports.

    Analyzes all available snapshots regardless of correctness results,
    including problems whose evaluation was skipped as already current.
    Writes per-checkpoint sidecars and the run-level aggregate. Returns
    a small summary dict, or None when nothing was measured.

    Failures are isolated here: unavailable package, invalid settings,
    and analyzer errors become logged diagnostics; never raises.
    """
    try:
        package = _require_package()
    except SloppinessUnavailableError as e:
        logger.error("Sloppiness measurement unavailable", error=str(e))
        return None

    analysis_settings = _build_settings(package, settings)
    if analysis_settings is None:
        return None

    try:
        report = package.analyze_run(
            run_dir, problems, settings=analysis_settings
        )
    except Exception as e:  # noqa: BLE001 - isolated by contract
        logger.error(
            "Sloppiness analysis failed",
            run_dir=str(run_dir),
            error=str(e),
            error_type=type(e).__name__,
        )
        return None

    _persist_run_report(run_dir, report, package)
    coverage = report.get("coverage", {})
    return {
        "enabled": True,
        "status": report.get("status"),
        "checkpoints_analyzed": coverage.get("checkpoints_analyzed", 0),
        "checkpoints_missing": coverage.get("checkpoints_missing", 0),
        "checkpoints_errors": coverage.get("checkpoints_errors", 0),
        "checkpoints_unsupported": coverage.get(
            "checkpoints_unsupported", 0
        ),
        "problems": {
            name: data.get("status")
            for name, data in (report.get("problems") or {}).items()
        },
    }


def measure_checkpoint(
    problem: Any,
    checkpoint: Any,
    checkpoint_dir: Path,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Measure one checkpoint snapshot against its predecessor.

    Used during runs, where checkpoints complete one at a time. The
    predecessor is the previous declared checkpoint when its snapshot
    exists; gaps are never bridged.

    Failures are isolated here: they are logged as diagnostics and
    never propagate to the caller.
    """
    try:
        package = _require_package()
    except SloppinessUnavailableError as e:
        logger.error("Sloppiness measurement unavailable", error=str(e))
        return None

    snapshot_dir = checkpoint_dir / "snapshot"
    if not snapshot_dir.is_dir():
        return None

    predecessor_root = _find_predecessor_snapshot(
        problem, checkpoint, checkpoint_dir
    )

    try:
        analysis_settings = _build_settings(package, settings)
        if analysis_settings is None:
            return None
        report = package.analyze_snapshot(
            snapshot_dir,
            predecessor_root=predecessor_root,
            subject=_subject(package, problem, checkpoint),
            settings=analysis_settings,
        )
    except Exception as e:  # noqa: BLE001 - isolated by contract
        logger.error(
            "Sloppiness checkpoint analysis failed",
            problem=problem.name,
            checkpoint=checkpoint.name,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None

    _atomic_write_json(sidecar_path(checkpoint_dir), report, package)
    return report


def checkpoint_fields(checkpoint_dir: Path) -> dict[str, Any]:
    """Flat ``sloppiness.*`` fields from a checkpoint sidecar.

    Works without the standalone package so that rebuilding summaries
    preserves previously persisted measurements. Returns {} when no
    valid sidecar exists.
    """
    path = sidecar_path(checkpoint_dir)
    if not path.is_file():
        return {}
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Failed to load sloppiness sidecar",
            path=str(path),
            error=str(e),
        )
        return {}
    return flatten_report(report)


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten a snapshot report to ``sloppiness.*`` fields.

    Mirrors ``sloppiness.report.flat_fields`` without importing the
    package (see equivalence tests).
    """
    out: dict[str, Any] = {
        "sloppiness.status": report.get("status"),
        "sloppiness.schema_version": report.get("schema_version"),
    }
    for _metric_id, record in (report.get("metrics") or {}).items():
        descriptor = record.get("descriptor") or {}
        namespace = descriptor.get("namespace")
        if not namespace:
            continue
        out[f"sloppiness.{namespace}.status"] = record.get("status")
        for name, value in (record.get("fields") or {}).items():
            key = f"sloppiness.{namespace}.{name}"
            out[key] = value
            if value is None and record.get("reason"):
                out[f"{key}.reason"] = record["reason"]
    for key, value in (report.get("deltas") or {}).items():
        out[f"sloppiness.delta.{key}"] = value
        reason = (report.get("delta_reasons") or {}).get(key)
        if value is None and reason:
            out[f"sloppiness.delta.{key}.reason"] = reason
    return out


def run_summary_fields(run_dir: Path) -> dict[str, Any]:
    """Compact sloppiness summary for the SCB run summary."""
    path = run_report_path(run_dir)
    if not path.is_file():
        return {}
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    coverage = report.get("coverage") or {}
    return {
        "enabled": True,
        "status": report.get("status"),
        "problems_measured": len(report.get("problems") or {}),
        "checkpoints_analyzed": coverage.get("checkpoints_analyzed", 0),
        "checkpoints_missing": coverage.get("checkpoints_missing", 0),
        "checkpoints_errors": coverage.get("checkpoints_errors", 0),
        "checkpoints_unsupported": coverage.get(
            "checkpoints_unsupported", 0
        ),
    }


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _build_settings(
    package: Any, settings: dict[str, Any] | None
) -> Any | None:
    """Build AnalysisSettings; invalid settings are logged, None returned."""
    if not settings:
        return package.AnalysisSettings()
    try:
        return package.settings_from_dict(settings)
    except Exception as e:  # noqa: BLE001 - isolated by contract
        logger.error(
            "Invalid sloppiness settings",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


def _subject(package: Any, problem: Any, checkpoint: Any) -> Any:
    return package.SubjectIdentity(
        problem=problem.name,
        checkpoint=checkpoint.name,
        order=checkpoint.order,
    )


def _find_predecessor_snapshot(
    problem: Any, checkpoint: Any, checkpoint_dir: Path
) -> Path | None:
    """Snapshot of the previous declared checkpoint, when available."""
    checkpoints = sorted(
        problem.checkpoints.values(), key=lambda c: c.order
    )
    previous = [
        c for c in checkpoints if c.order < checkpoint.order
    ]
    if not previous:
        return None
    prev_checkpoint = previous[-1]
    candidate = (
        checkpoint_dir.parent / prev_checkpoint.name / "snapshot"
    )
    return candidate if candidate.is_dir() else None


def _persist_run_report(run_dir: Path, report: dict, package: Any) -> None:
    _atomic_write_json(run_report_path(run_dir), report, package)
    # Per-checkpoint sidecars for freshly measured problems.
    for problem_name, problem_report in (report.get("problems") or {}).items():
        for checkpoint_name, cp_report in (
            problem_report.get("checkpoints") or {}
        ).items():
            if cp_report.get("status") == "missing":
                continue
            sidecar = sidecar_path(run_dir / problem_name / checkpoint_name)
            _atomic_write_json(sidecar, cp_report, package)


def _atomic_write_json(path: Path, payload: dict, package: Any) -> None:
    """Atomic write so concurrent workers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(package.to_json(payload))
        tmp.replace(path)
    except OSError as e:
        logger.warning(
            "Failed to persist sloppiness report",
            path=str(path),
            error=str(e),
        )
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
