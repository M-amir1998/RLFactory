from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from rlfactory.cli.app import app
from rlfactory.config.loader import load_model
from rlfactory.config.settings import Settings
from rlfactory.errors import ExperimentError
from rlfactory.experiments.config import STAGE_ORDER, ExperimentConfig, stage_dependencies
from rlfactory.experiments.runner import ExperimentRunner, code_fingerprint

RUNNER = CliRunner()


def _config(config_dir: Path, **updates: object) -> ExperimentConfig:
    base = load_model(config_dir / "experiments" / "ci-e2e.yaml", ExperimentConfig)
    return ExperimentConfig.model_validate({**base.model_dump(mode="json"), **updates})


def test_stage_dependencies_respect_training_method(config_dir: Path) -> None:
    config = _config(config_dir)
    assert stage_dependencies("train", config) == ("build_sft",)
    assert stage_dependencies("gate", config) == ("evaluate_base", "evaluate_candidate")
    grpo = _config(
        config_dir,
        training={
            **config.training.model_dump(mode="json"),  # type: ignore[union-attr]
            "method": "grpo",
            "config": {"method_params": {"method": "grpo", "num_generations": 2}},
        },
    )
    assert stage_dependencies("train", grpo) == ("build_grpo",)


def test_experiment_config_validation(config_dir: Path) -> None:
    with pytest.raises(ValidationError, match="depends on"):
        _config(config_dir, stages=["rollouts"])
    with pytest.raises(ValidationError, match="training section"):
        _config(config_dir, training=None)
    with pytest.raises(ValidationError, match="slug"):
        _config(config_dir, name="Bad Name")
    with pytest.raises(ValidationError, match="method_params"):
        _config(config_dir, training={"backend": "mock", "method": "dpo", "model": {"name": "m"}})
    assert _config(config_dir, stages=["tasks", "evaluate_base", "report"]).stages == (
        "tasks",
        "evaluate_base",
        "report",
    )


def test_all_repo_experiment_configs_validate(config_dir: Path) -> None:
    configs = sorted((config_dir / "experiments").glob("*.yaml"))
    assert len(configs) >= 5
    for path in configs:
        assert load_model(path, ExperimentConfig).name


def test_plan_counts_and_requirements(config_dir: Path, tmp_path: Path) -> None:
    runner = ExperimentRunner(Settings(config_dir=config_dir, results_dir=tmp_path))
    plan = runner.plan(_config(config_dir))
    assert [s.name for s in plan.stages] == list(STAGE_ORDER)
    assert plan.rollout_trajectories == plan.train_tasks * 4 and plan.simulated
    assert plan.external_requirements == ("none (CPU-only, offline)",)
    gpu = runner.plan(
        load_model(config_dir / "experiments" / "gpu-qwen3-grpo-binary.yaml", ExperimentConfig)
    )
    assert not gpu.simulated and any("GPU host" in r for r in gpu.external_requirements)


def test_create_run_refuses_duplicates_and_bad_selections(config_dir: Path, tmp_path: Path) -> None:
    import asyncio

    runner = ExperimentRunner(Settings(config_dir=config_dir, results_dir=tmp_path))
    run_dir = runner.create_run(_config(config_dir), "dup")
    assert (run_dir / "manifest.json").exists() and (run_dir / "config.yaml").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["simulated"] is True and manifest["task_suites"][0]["name"] == "ci-tiny"
    with pytest.raises(ExperimentError, match="already exists"):
        runner.create_run(_config(config_dir), "dup")
    with pytest.raises(ExperimentError, match="unknown stages"):
        asyncio.run(runner.run(run_dir, only=["bogus"]))
    with pytest.raises(ExperimentError, match="needs completed"):
        asyncio.run(runner.run(run_dir, only=["rollouts"]))
    assert len(code_fingerprint()) == 16


def test_cli_basics(config_dir: Path, tmp_path: Path) -> None:
    common = ["--config-dir", str(config_dir), "--results-dir", str(tmp_path)]
    assert RUNNER.invoke(app, ["version"]).exit_code == 0
    listing = RUNNER.invoke(app, [*common, "tasks", "list"])
    assert listing.exit_code == 0 and "ci-tiny" in listing.stdout
    inspect = RUNNER.invoke(app, [*common, "tasks", "inspect", "ci-tiny", "--limit", "1"])
    assert inspect.exit_code == 0 and "families:" in inspect.stdout
    envs = RUNNER.invoke(app, ["environments", "list"])
    assert envs.exit_code == 0 and "tool_use" in envs.stdout
    plan = RUNNER.invoke(
        app, [*common, "experiment", "plan", str(config_dir / "experiments" / "ci-e2e.yaml")]
    )
    assert plan.exit_code == 0 and "rollout trajectories" in plan.stdout
    doctor = RUNNER.invoke(app, [*common, "doctor", "--no-docker", "--json"])
    assert doctor.exit_code == 0 and '"name": "python"' in doctor.stdout


def test_cli_environment_check_and_dry_runs(config_dir: Path, tmp_path: Path) -> None:
    common = ["--config-dir", str(config_dir), "--results-dir", str(tmp_path)]
    check = RUNNER.invoke(app, ["environments", "check", "transformation", "--samples", "2"])
    assert check.exit_code == 0, check.stdout
    dry = RUNNER.invoke(
        app,
        [
            *common,
            "rollout",
            "run",
            str(config_dir / "rollouts" / "mock-train-mixed.yaml"),
            "--dry-run",
        ],
    )
    assert dry.exit_code == 0 and "dry run" in dry.stdout
    experiment_dry = RUNNER.invoke(
        app,
        [
            *common,
            "experiment",
            "run",
            str(config_dir / "experiments" / "ci-e2e.yaml"),
            "--dry-run",
        ],
    )
    assert experiment_dry.exit_code == 0 and not list(tmp_path.iterdir())


def test_cli_errors_are_reported(config_dir: Path, tmp_path: Path) -> None:
    common = ["--config-dir", str(config_dir), "--results-dir", str(tmp_path)]
    missing = RUNNER.invoke(app, [*common, "experiment", "resume", "nope"])
    assert missing.exit_code == 1 and "no experiment run" in missing.stdout
    report = RUNNER.invoke(app, [*common, "report", "nope"])
    assert report.exit_code == 1
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": []}\n', encoding="utf-8")
    validate = RUNNER.invoke(app, ["dataset", "validate", str(bad), "--kind", "sft"])
    assert validate.exit_code == 1


def test_cli_models_plan_tasks_sample_and_lm_eval(config_dir: Path, tmp_path: Path) -> None:
    common = ["--config-dir", str(config_dir), "--results-dir", str(tmp_path)]
    refused = RUNNER.invoke(app, [*common, "models", "plan", "org/not-cached-model"])
    assert refused.exit_code == 1 and "--allow-download" in refused.stdout
    out = tmp_path / "sample.jsonl"
    sampled = RUNNER.invoke(
        app,
        [
            *common,
            "tasks",
            "sample",
            "ci-tiny",
            "--n",
            "5",
            "--out",
            str(out),
            "--strategy",
            "difficulty_balanced",
        ],
    )
    assert sampled.exit_code == 0 and len(out.read_text().splitlines()) == 5
    command = RUNNER.invoke(
        app, ["external", "lm-eval", "--model", "m", "--base-url", "http://h:8000"]
    )
    assert command.exit_code == 0 and "lm_eval --model local-chat-completions" in command.stdout
    dry = RUNNER.invoke(
        app, [*common, "evaluate", str(config_dir / "evaluation" / "mock-base.yaml"), "--dry-run"]
    )
    assert dry.exit_code == 0 and "dry run" in dry.stdout


def test_cli_remote_launch_dry_run_never_touches_the_handle(tmp_path: Path) -> None:
    from rlfactory.domain.model import ModelReference
    from rlfactory.domain.training import (
        DatasetReference,
        LaunchMode,
        PreparedTrainingRun,
        TrainingMethod,
        TrainingRun,
    )
    from rlfactory.training.base import save_prepared

    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "run_config.json").write_text(
        json.dumps({"config_kwargs": {"max_steps": -1}}), encoding="utf-8"
    )
    run = TrainingRun(
        run_id="r1",
        method=TrainingMethod.SFT,
        model=ModelReference(name="Qwen/Qwen2.5-0.5B", revision="a" * 40),
        dataset=DatasetReference(
            name="d", path="p", schema_name="s", num_records=1, fingerprint="f"
        ),
        launch_mode=LaunchMode.REMOTE,
    )
    save_prepared(
        PreparedTrainingRun(
            run=run, backend="trl", workdir=str(workdir), command=("bash", "launch.sh")
        )
    )
    host = tmp_path / "host.yaml"
    host.write_text(
        "name: box\nhost: gpu-alias\nworkspace: /srv/runs\ncode_dir: /srv/RLFactory\n",
        encoding="utf-8",
    )
    # A handle from an earlier real launch must survive a later dry run untouched.
    real_handle = '{"handle_id": "real-job"}'
    (workdir / "remote_handle.json").write_text(real_handle, encoding="utf-8")
    result = RUNNER.invoke(
        app, ["remote", "launch", str(workdir), "--host", str(host), "--dry-run"]
    )
    assert result.exit_code == 0, result.stdout
    assert "ssh" in result.stdout and "nothing was launched" in result.stdout
    assert "launched:" not in result.stdout
    assert (workdir / "remote_handle.json").read_text(encoding="utf-8") == real_handle
