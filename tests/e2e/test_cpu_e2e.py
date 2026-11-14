"""Full CPU-only post-training E2E: tasks -> rollouts -> rewards -> SFT/DPO/GRPO datasets ->
mock training -> evaluation -> regression gate -> analysis -> flywheel -> report. No GPU, no network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pyarrow.parquet as pq

from rlfactory.config.loader import load_model
from rlfactory.config.settings import Settings
from rlfactory.datasets.validation import validate_trl_file
from rlfactory.experiments.config import STAGE_ORDER, ExperimentConfig
from rlfactory.experiments.runner import ExperimentRunner


def test_ci_experiment_end_to_end_and_resume(config_dir: Path, tmp_path: Path) -> None:
    settings = Settings(config_dir=config_dir, results_dir=tmp_path)
    runner = ExperimentRunner(settings)
    config = load_model(config_dir / "experiments" / "ci-e2e.yaml", ExperimentConfig)
    run_dir = runner.create_run(config, "ci-e2e-test")

    records = asyncio.run(runner.run(run_dir))
    assert [r.name for r in records] == list(STAGE_ORDER)
    assert all(r.status == "completed" and not r.cached for r in records)

    for kind in ("sft", "dpo", "grpo"):
        assert validate_trl_file(run_dir / "datasets" / kind / "train.jsonl", kind).ok  # type: ignore[arg-type]
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    base, candidate = evaluation["baseline"], evaluation["candidate"]
    assert base["simulated"] and candidate["simulated"]
    assert (
        candidate["overall"]["verified_accuracy"] >= base["overall"]["verified_accuracy"]
    )  # common random numbers
    assert evaluation["promotion"]["state"] in {"validated", "rejected"}
    assert (tmp_path / "model_registry.json").exists()
    assert pq.read_table(run_dir / "failures.parquet").num_rows > 0
    metrics = (run_dir / "metrics.prom").read_text(encoding="utf-8")
    assert (
        "rlfactory_trajectories_total" in metrics
        and "rlfactory_policy_latency_seconds_bucket" in metrics
    )
    flywheel = json.loads((run_dir / "flywheel.json").read_text())
    assert flywheel[0]["eval_accuracy"] == base["overall"]["verified_accuracy"]
    assert len(flywheel) == 2 and flywheel[1]["tasks_added"] >= 0

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## Experiment",
        "## Tasks",
        "## Rollouts",
        "## Reward",
        "## Datasets",
        "## Training",
        "## Evaluation",
        "## Regression gate",
        "## Failure analysis",
        "## Failure-driven data flywheel",
        "## Resource usage",
        "## Limitations",
    ):
        assert heading in report
    assert "SIMULATED RESULTS" in report
    assert (
        "| Model | Method | Verified Acc. | Python Pass | Format | Reward | Train Time | VRAM |"
        in report
    )
    assert {p.name for p in (run_dir / "plots").glob("*.svg")} >= {
        "accuracy_by_domain.svg",
        "reward_distribution.svg",
    }
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for key in (
        "run_id",
        "code",
        "seed",
        "task_suites",
        "environments",
        "verifiers",
        "policy",
        "generation",
        "training_backend",
        "hardware",
        "config_fingerprint",
    ):
        assert key in manifest

    resumed = asyncio.run(runner.run(run_dir))
    assert all(r.cached for r in resumed if r.name != "report")

    forced = asyncio.run(runner.run(run_dir, only=["evaluate_base"], force=["evaluate_base"]))
    assert len(forced) == 1 and not forced[0].cached
