"""Tests for the sloppiness adapter and integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slop_code.metrics.summary.compute import compute_run_summary
from slop_code.sloppiness import flatten_report
from slop_code.sloppiness import is_available
from slop_code.sloppiness import measure_run
from slop_code.sloppiness import run_report_path
from slop_code.sloppiness import run_summary_fields
from slop_code.sloppiness import sidecar_path

sloppiness = pytest.importorskip(
    "sloppiness", reason="standalone package not installed"
)

CP1_CODE = "def main():\n    return 1\n"
CP2_CODE = "import os\n\n\ndef main():\n    return os.getpid()\n"
CP3_CODE = (
    "import os\n\n\ndef a():\n    return os.getpid()\n\n\n"
    "def main():\n    return a()\n"
)


def make_snapshot(root: Path, files: dict[str, str]) -> None:
    root.mkdir(parents=True)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def make_problem_run(
    run_dir: Path,
    name: str = "tiny",
    checkpoints: int = 3,
) -> Path:
    problem_dir = run_dir / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "problem.yaml").write_text(
        "checkpoints:\n"
        + "".join(
            f"  checkpoint_{i}:\n    order: {i}\n"
            for i in range(1, checkpoints + 1)
        )
    )
    code = [
        ("checkpoint_1", CP1_CODE),
        ("checkpoint_2", CP2_CODE),
        ("checkpoint_3", CP3_CODE),
    ]
    for checkpoint_name, content in code[: checkpoints]:
        make_snapshot(
            problem_dir / checkpoint_name / "snapshot",
            {"main.py": content},
        )
    return problem_dir


class TestMeasureRun:
    def test_writes_sidecars_and_run_report(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        make_problem_run(run_dir)

        summary = measure_run(run_dir)

        assert summary is not None
        assert summary["checkpoints_analyzed"] == 3
        # The first checkpoint has no baseline for change metrics, so
        # the run status is partial, not ok.
        assert summary["status"] == "partial"
        assert summary["problems"] == {"tiny": "partial"}

        report = json.loads(run_report_path(run_dir).read_text())
        assert report["schema_version"] == 1
        sidecar = sidecar_path(run_dir / "tiny" / "checkpoint_2")
        assert sidecar.is_file()
        cp_report = json.loads(sidecar.read_text())
        assert cp_report["metrics"]["import_graph"]["status"] == "ok"
        statuses = {
            name: cp["status"]
            for name, cp in report["problems"]["tiny"]["checkpoints"].items()
        }
        assert statuses == {
            "checkpoint_1": "partial",
            "checkpoint_2": "ok",
            "checkpoint_3": "ok",
        }

    def test_first_checkpoint_has_no_change_baseline(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        make_problem_run(run_dir)

        measure_run(run_dir)

        cp1 = json.loads(
            sidecar_path(run_dir / "tiny" / "checkpoint_1").read_text()
        )
        assert cp1["metrics"]["change_dispersion"]["status"] == "unavailable"
        assert cp1["metrics"]["change_dispersion"]["reason"] == "no_baseline"
        cp2 = json.loads(
            sidecar_path(run_dir / "tiny" / "checkpoint_2").read_text()
        )
        assert cp2["metrics"]["change_dispersion"]["status"] == "ok"

    def test_missing_predecessor_snapshot_is_not_bridged(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        make_problem_run(run_dir, checkpoints=3)
        # checkpoint_2 snapshot removed: checkpoint_3 measures against
        # nothing rather than bridging to checkpoint_1.
        import shutil

        shutil.rmtree(run_dir / "tiny" / "checkpoint_2" / "snapshot")

        measure_run(run_dir)

        cp3 = json.loads(
            sidecar_path(run_dir / "tiny" / "checkpoint_3").read_text()
        )
        assert cp3["metrics"]["change_dispersion"]["status"] == "unavailable"
        assert (
            cp3["metrics"]["change_dispersion"]["reason"]
            == "missing_predecessor"
        )
    def test_problems_filter_limits_measurement(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        make_problem_run(run_dir, name="alpha")
        make_problem_run(run_dir, name="beta")

        measure_run(run_dir, problems=["alpha"])

        assert sidecar_path(run_dir / "alpha" / "checkpoint_1").is_file()
        assert not sidecar_path(run_dir / "beta" / "checkpoint_1").exists()

    def test_problem_without_checkpoint_dirs_is_absent(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        problem = run_dir / "empty"
        problem.mkdir(parents=True)
        (problem / "problem.yaml").write_text(
            "checkpoints:\n  checkpoint_1:\n    order: 1\n"
        )

        summary = measure_run(run_dir)

        # Discovery only finds problems that produced checkpoints; the
        # agent never got anywhere here, so nothing is measured.
        assert summary["problems"] == {}
        assert not run_report_path(run_dir).exists() or json.loads(
            run_report_path(run_dir).read_text()
        )["problems"] == {}

    def test_run_summary_fields(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        make_problem_run(run_dir)
        assert run_summary_fields(run_dir) == {}

        measure_run(run_dir)

        summary = run_summary_fields(run_dir)
        assert summary["enabled"] is True
        assert summary["checkpoints_analyzed"] == 3
        assert summary["problems_measured"] == 1


class TestFlattenReport:
    def test_matches_package_flat_fields(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        make_problem_run(run_dir)
        measure_run(run_dir)

        for checkpoint in ("checkpoint_1", "checkpoint_2", "checkpoint_3"):
            report = json.loads(
                sidecar_path(run_dir / "tiny" / checkpoint).read_text()
            )
            assert flatten_report(report) == sloppiness.flat_fields(report)

    def test_namespaced_fields_and_reasons(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        make_problem_run(run_dir)
        measure_run(run_dir)

        fields = flatten_report(
            json.loads(
                sidecar_path(run_dir / "tiny" / "checkpoint_2").read_text()
            )
        )

        assert fields["sloppiness.status"] == "ok"
        assert fields["sloppiness.imports.status"] == "ok"
        assert "sloppiness.imports.dependency_top_q_share" in fields
        assert fields["sloppiness.calls.status"] == "ok"
        assert fields["sloppiness.change.status"] == "ok"
        assert fields["sloppiness.change.total_churn"] > 0

    def test_unavailable_field_carries_reason(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        make_problem_run(run_dir)
        measure_run(run_dir)

        fields = flatten_report(
            json.loads(
                sidecar_path(run_dir / "tiny" / "checkpoint_1").read_text()
            )
        )

        assert fields["sloppiness.change.normalized_entropy"] is None
        reason_key = "sloppiness.change.normalized_entropy.reason"
        assert fields[reason_key] == "no_baseline"


class TestCheckpointFields:
    def test_reads_sidecar_without_package(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # checkpoint_fields must work when the standalone package is
        # absent (summary rebuilds preserve measurements).
        run_dir = tmp_path / "run"
        make_problem_run(run_dir)
        measure_run(run_dir)
        checkpoint_dir = run_dir / "tiny" / "checkpoint_1"

        import builtins

        real_import = builtins.__import__

        def no_sloppiness(name, *args, **kwargs):  # noqa: ANN002, ANN003
            if name.split(".")[0] == "sloppiness":
                raise ImportError("sloppiness not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_sloppiness)

        from slop_code.sloppiness import checkpoint_fields

        fields = checkpoint_fields(checkpoint_dir)
        assert fields["sloppiness.status"] in {"ok", "partial"}
        assert "sloppiness.imports.dependency_top_q_share" in fields

    def test_missing_sidecar_returns_empty(self, tmp_path: Path) -> None:
        from slop_code.sloppiness import checkpoint_fields

        assert checkpoint_fields(tmp_path) == {}

    def test_corrupt_sidecar_returns_empty(self, tmp_path: Path) -> None:
        from slop_code.sloppiness import checkpoint_fields

        sidecar = sidecar_path(tmp_path)
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text("{not json")
        assert checkpoint_fields(tmp_path) == {}


class TestGetCheckpointMetrics:
    def test_merges_sloppiness_fields(self, tmp_path: Path) -> None:
        from slop_code.metrics.checkpoint.driver import get_checkpoint_metrics

        run_dir = tmp_path / "run"
        make_problem_run(run_dir)
        measure_run(run_dir)

        metrics = get_checkpoint_metrics(run_dir / "tiny" / "checkpoint_2")

        assert "sloppiness.status" in metrics
        assert "sloppiness.imports.dependency_top_q_share" in metrics
        assert "sloppiness.change.total_churn" in metrics

    def test_no_sidecar_no_sloppiness_keys(self, tmp_path: Path) -> None:
        from slop_code.metrics.checkpoint.driver import get_checkpoint_metrics

        (tmp_path / "checkpoint_1").mkdir()
        metrics = get_checkpoint_metrics(tmp_path / "checkpoint_1")

        assert not [k for k in metrics if k.startswith("sloppiness.")]


class TestMeasureCheckpoint:
    """Runner-path measurement of one checkpoint against its predecessor."""

    @staticmethod
    def _make_problem():
        from types import SimpleNamespace

        return SimpleNamespace(
            name="tiny",
            checkpoints={
                f"checkpoint_{i}": SimpleNamespace(
                    name=f"checkpoint_{i}", order=i
                )
                for i in (1, 2, 3)
            },
        )

    def _make_checkpoint_dir(self, run: Path, name: str, code: str) -> Path:
        d = run / "tiny" / name
        (d / "snapshot").mkdir(parents=True, exist_ok=True)
        (d / "snapshot" / "main.py").write_text(code)
        return d

    def test_measures_against_predecessor_snapshot(
        self, tmp_path: Path
    ) -> None:
        from slop_code.sloppiness import measure_checkpoint

        run = tmp_path / "run"
        self._make_checkpoint_dir(
            run,
            "checkpoint_1",
            "def main():\n    return 1\n"
        )
        cp2 = self._make_checkpoint_dir(run, "checkpoint_2", CP2_CODE)
        problem = self._make_problem()

        report = measure_checkpoint(
            problem=problem,
            checkpoint=problem.checkpoints["checkpoint_2"],
            checkpoint_dir=cp2,
        )

        assert report is not None
        assert report["metrics"]["change_dispersion"]["status"] == "ok"
        assert sidecar_path(cp2).is_file()
        # Sidecar matches what measure_run would write for this checkpoint.
        assert report["subject"]["problem"] == "tiny"
        assert report["subject"]["checkpoint"] == "checkpoint_2"

    def test_first_checkpoint_has_no_predecessor(self, tmp_path: Path) -> None:
        from slop_code.sloppiness import measure_checkpoint

        run = tmp_path / "run"
        cp1 = self._make_checkpoint_dir(
            run,
            "checkpoint_1",
            "def main():\n    return 1\n"
        )
        problem = self._make_problem()

        report = measure_checkpoint(
            problem=problem,
            checkpoint=problem.checkpoints["checkpoint_1"],
            checkpoint_dir=cp1,
        )

        assert report is not None
        assert (
            report["metrics"]["change_dispersion"]["reason"] == "no_baseline"
        )

    def test_missing_predecessor_snapshot_is_not_bridged(
        self, tmp_path: Path
    ) -> None:
        from slop_code.sloppiness import measure_checkpoint

        run = tmp_path / "run"
        self._make_checkpoint_dir(run, "checkpoint_1", CP1_CODE)
        # checkpoint_2 has a directory but no snapshot
        (run / "tiny" / "checkpoint_2").mkdir(parents=True)
        cp3 = self._make_checkpoint_dir(run, "checkpoint_3", CP2_CODE)
        problem = self._make_problem()

        report = measure_checkpoint(
            problem=problem,
            checkpoint=problem.checkpoints["checkpoint_3"],
            checkpoint_dir=cp3,
        )

        assert report is not None
        assert (
            report["metrics"]["change_dispersion"]["reason"] == "no_baseline"
        )

    def test_no_snapshot_returns_none(self, tmp_path: Path) -> None:
        from slop_code.sloppiness import measure_checkpoint

        run = tmp_path / "run"
        cp1 = run / "tiny" / "checkpoint_1"
        cp1.mkdir(parents=True)
        problem = self._make_problem()

        report = measure_checkpoint(
            problem=problem,
            checkpoint=problem.checkpoints["checkpoint_1"],
            checkpoint_dir=cp1,
        )

        assert report is None

    def test_failure_isolated_returns_none(self, tmp_path: Path) -> None:
        from slop_code.sloppiness import measure_checkpoint

        run = tmp_path / "run"
        cp1 = self._make_checkpoint_dir(
            run,
            "checkpoint_1",
            "def main():\n    return 1\n"
        )
        problem = self._make_problem()

        report = measure_checkpoint(
            problem=problem,
            checkpoint=problem.checkpoints["checkpoint_1"],
            checkpoint_dir=cp1,
            settings={"unknown_option": 1},
        )

        # Invalid settings are isolated: no report, no crash.
        assert report is None


class TestConfigPlumbing:
    def test_disabled_by_default(self) -> None:
        from slop_code.entrypoints.config.run_config import RunConfig
        from slop_code.entrypoints.config.run_config import (
            SloppinessRunSettings,
        )

        cfg = RunConfig.model_validate({})
        assert cfg.sloppiness.enabled is False
        assert SloppinessRunSettings().enabled is False

    def test_parses_enabled_and_settings(self) -> None:
        from slop_code.entrypoints.config.run_config import RunConfig

        cfg = RunConfig.model_validate(
            {
                "sloppiness": {
                    "enabled": True,
                    "settings": {"q": 0.1, "include_tests": True},
                }
            }
        )
        assert cfg.sloppiness.enabled is True
        assert cfg.sloppiness.settings == {"q": 0.1, "include_tests": True}

    def test_rejects_unknown_keys(self) -> None:
        from pydantic import ValidationError

        from slop_code.entrypoints.config.run_config import RunConfig

        with pytest.raises(ValidationError, match="sloppiness"):
            RunConfig.model_validate(
                {"sloppiness": {"enabled": True, "nope": 1}}
            )

    def test_task_config_carries_sloppiness(self, tmp_path: Path) -> None:
        from slop_code.entrypoints.commands.run_agent import _create_task_config
        from slop_code.entrypoints.problem_runner.models import RunTaskConfig

        base = RunTaskConfig(
            problem_base_path=tmp_path,
            run_dir=tmp_path / "run",
            env_spec=None,
            agent_config=None,
            model_def=None,
            credential=None,
            prompt_template="",
            pass_policy=None,
            seed=0,
            verbosity=0,
            image="img",
        )
        assert base.sloppiness is None

        cfg = _create_task_config(
            problem_base_path=tmp_path,
            run_dir=tmp_path / "run",
            env_spec=None,
            agent_config=None,
            model_def=None,
            credential=None,
            run_cfg=_make_run_cfg(enabled=True),
            seed=0,
            verbosity=0,
            debug=False,
            evaluate=True,
            live_progress=False,
            image_name="img",
            resume=False,
            concurrent_evaluation=False,
        )
        assert cfg.sloppiness is not None
        assert cfg.sloppiness.enabled is True


def _make_run_cfg(*, enabled: bool):
    from slop_code.entrypoints.config.run_config import ResolvedRunConfig
    from slop_code.entrypoints.config.run_config import SloppinessRunSettings
    from slop_code.evaluation import PassPolicy as PassPolicyEnum

    return ResolvedRunConfig(
        agent_config_path=None,
        agent={},
        environment_config_path=None,
        environment={},
        prompt_path="p.jinja",
        prompt_content="",
        model={"provider": "x", "name": "m"},
        thinking="none",
        thinking_max_tokens=None,
        pass_policy=PassPolicyEnum.ANY_CASE,
        problems=[],
        save_dir="outputs",
        save_template="x",
        output_path="outputs/x",
        one_shot={},
        sloppiness=SloppinessRunSettings(enabled=enabled),
    )


class TestEvalCommand:
    def test_measure_sloppiness_wrapper_isolates_unavailable(
        self, tmp_path
    ) -> None:
        # _measure_sloppiness never raises; unavailable package is an
        # error log, not an eval failure.
        from slop_code.entrypoints.commands.eval_run_dir import (
            _measure_sloppiness,
        )

        _measure_sloppiness(tmp_path)  # empty dir: no problems, no crash


class TestRunSummarySloppiness:
    @pytest.fixture
    def mock_config(self) -> dict:
        return {
            "model": {"name": "test-model"},
            "thinking": "none",
            "prompt_path": "test.jinja",
            "agent": {"type": "test-agent", "version": "1.0"},
        }

    def test_summary_includes_sloppiness_when_measured(
        self, tmp_path: Path, mock_config
    ) -> None:
        run_dir = tmp_path / "run"
        make_problem_run(run_dir)
        measure_run(run_dir)

        checkpoints = [
            {
                "problem": "tiny",
                "idx": 1,
                "sloppiness.status": "partial",
                "sloppiness.imports.dependency_top_q_share": 0.0,
            },
            {
                "problem": "tiny",
                "idx": 2,
                "sloppiness.status": "ok",
                "sloppiness.change.total_churn": 10,
            },
        ]
        summary = compute_run_summary(
            mock_config, checkpoints, expected_checkpoints=2, run_dir=run_dir
        )

        assert summary.sloppiness is not None
        assert summary.sloppiness["enabled"] is True
        assert summary.sloppiness["checkpoints_analyzed"] == 3
        assert summary.sloppiness["checkpoints_with_measurements"] == 2
        assert summary.sloppiness["problems_with_measurements"] == 1

    def test_summary_without_sloppiness(self, mock_config) -> None:
        checkpoints = [{"problem": "tiny", "idx": 1}]
        summary = compute_run_summary(
            mock_config, checkpoints, expected_checkpoints=1, run_dir=None
        )
        assert summary.sloppiness is None

    def test_summary_serializes_sloppiness(
        self, tmp_path: Path, mock_config
    ) -> None:
        import json

        run_dir = tmp_path / "run"
        make_problem_run(run_dir)
        measure_run(run_dir)
        checkpoints = [
            {
                "problem": "tiny",
                "idx": 1,
                "sloppiness.status": "partial",
            }
        ]
        summary = compute_run_summary(
            mock_config, checkpoints, expected_checkpoints=1, run_dir=run_dir
        )

        data = json.loads(summary.model_dump_json())
        assert data["sloppiness"]["enabled"] is True
        assert data["sloppiness"]["status"] == "partial"


class TestAvailability:
    def test_is_available_when_installed(self) -> None:
        assert is_available() is True

    def test_measure_run_isolated_when_package_missing(
        self, monkeypatch
    ) -> None:
        # Unavailable package is a logged diagnostic, not a crash.
        import builtins
        from pathlib import Path as _Path

        real_import = builtins.__import__

        def no_sloppiness(name, *args, **kwargs):  # noqa: ANN002, ANN003
            if name.split(".")[0] == "sloppiness":
                raise ImportError("sloppiness not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_sloppiness)

        assert measure_run(_Path.cwd()) is None


class TestRunnerHook:
    """Opt-in measurement at the evaluate_agent_snapshot seam."""

    @staticmethod
    def _stub_problem():
        from types import SimpleNamespace

        return SimpleNamespace(
            name="tiny",
            entry_file="main.py",
            checkpoints={
                "checkpoint_1": SimpleNamespace(name="checkpoint_1", order=1)
            },
        )

    def _patch_seams(self, monkeypatch):
        import slop_code.agent_runner.runner as runner

        calls = {"measure": 0}

        monkeypatch.setattr(
            runner,
            "evaluate_checkpoint",
            lambda **kwargs: type(
                "R",
                (),
                {
                    "save": lambda self, d: None,
                    "pass_counts": {},
                    "total_counts": {},
                },
            )(),
        )
        monkeypatch.setattr(
            runner,
            "measure_snapshot_quality",
            lambda entry, snapshot: (None, []),
        )
        monkeypatch.setattr(
            runner, "save_quality_metrics", lambda *a: None
        )
        monkeypatch.setattr(
            runner.SnapshotQualityReport,
            "from_snapshot_metrics",
            classmethod(lambda cls, m: object.__new__(cls)),
        )

        import slop_code.sloppiness as adapter

        def fail_if_called(**kwargs):
            calls["measure"] += 1

        monkeypatch.setattr(
            adapter, "measure_checkpoint", fail_if_called, raising=False
        )
        return calls

    @staticmethod
    def _stub_environment():
        from types import SimpleNamespace

        return SimpleNamespace(format_entry_file=lambda entry: entry)

    def _run_hook(self, tmp_path: Path, sloppiness):
        from slop_code.agent_runner.runner import evaluate_agent_snapshot

        snapshot_dir = tmp_path / "checkpoint_1" / "snapshot"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "main.py").write_text(CP1_CODE)
        problem = self._stub_problem()
        return evaluate_agent_snapshot(
            checkpoint=problem.checkpoints["checkpoint_1"],
            save_dir=tmp_path / "checkpoint_1",
            snapshot_dir=snapshot_dir,
            problem=problem,
            environment=self._stub_environment(),
            sloppiness=sloppiness,
        )

    def test_disabled_does_not_measure(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from slop_code.entrypoints.config.run_config import (
            SloppinessRunSettings,
        )

        calls = self._patch_seams(monkeypatch)
        self._run_hook(tmp_path, SloppinessRunSettings(enabled=False))
        assert calls["measure"] == 0

    def test_none_does_not_measure(self, tmp_path: Path, monkeypatch) -> None:
        calls = self._patch_seams(monkeypatch)
        self._run_hook(tmp_path, None)
        assert calls["measure"] == 0

    def test_enabled_measures_and_writes_sidecar(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Route through the real adapter: the hook must produce a sidecar.
        import slop_code.agent_runner.runner as runner
        from slop_code.entrypoints.config.run_config import (
            SloppinessRunSettings,
        )

        monkeypatch.setattr(
            runner,
            "evaluate_checkpoint",
            lambda **kwargs: type(
                "R",
                (),
                {
                    "save": lambda self, d: None,
                    "pass_counts": {},
                    "total_counts": {},
                },
            )(),
        )
        monkeypatch.setattr(
            runner,
            "measure_snapshot_quality",
            lambda entry, snapshot: (None, []),
        )
        monkeypatch.setattr(
            runner, "save_quality_metrics", lambda *a: None
        )
        monkeypatch.setattr(
            runner.SnapshotQualityReport,
            "from_snapshot_metrics",
            classmethod(lambda cls, m: object.__new__(cls)),
        )

        self._run_hook(tmp_path, SloppinessRunSettings(enabled=True))

        sidecar = sidecar_path(tmp_path / "checkpoint_1")
        assert sidecar.is_file()
        fields = flatten_report(json.loads(sidecar.read_text()))
        assert fields["sloppiness.status"] in {"ok", "partial"}
        assert "sloppiness.imports.dependency_top_q_share" in fields

    def test_adapter_failure_does_not_break_evaluation(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import slop_code.sloppiness as adapter
        from slop_code.entrypoints.config.run_config import (
            SloppinessRunSettings,
        )

        def explode(**kwargs):
            raise RuntimeError("analyzer exploded")

        monkeypatch.setattr(
            adapter, "measure_checkpoint", explode, raising=False
        )

        # _measure_sloppiness in the runner isolates adapter exceptions;
        # evaluation itself must still succeed.
        self._patch_seams(monkeypatch)
        problem = self._stub_problem()
        from slop_code.agent_runner.runner import _measure_sloppiness

        _measure_sloppiness(
            problem,
            problem.checkpoints["checkpoint_1"],
            tmp_path / "checkpoint_1",
            SloppinessRunSettings(enabled=True),
        )  # must not raise


class TestRepeatedAggregation:
    def test_repeated_reads_are_stable_and_preserved(
        self, tmp_path: Path
    ) -> None:
        from slop_code.metrics.checkpoint.driver import get_checkpoint_metrics

        run_dir = tmp_path / "run"
        make_problem_run(run_dir)
        measure_run(run_dir)
        checkpoint_dir = run_dir / "tiny" / "checkpoint_2"

        first = get_checkpoint_metrics(checkpoint_dir)
        second = get_checkpoint_metrics(checkpoint_dir)
        baseline = get_checkpoint_metrics(run_dir / "tiny" / "checkpoint_1")

        assert first == second
        assert "sloppiness.imports.dependency_top_q_share" in first
        assert "sloppiness.change.total_churn" in first
        # Reading one checkpoint never disturbs another's sidecar.
        assert "sloppiness.status" in baseline
        # Rebuilding summaries does not erase persisted reports.
        assert sidecar_path(checkpoint_dir).is_file()
