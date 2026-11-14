"""Mock rollout E2E through the CLI: run, inspect, query, interrupt-free resume."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from rlfactory.cli.app import app

RUNNER = CliRunner()


def test_cli_rollout_run_inspect_query_and_resume(config_dir: Path, tmp_path: Path) -> None:
    job = {
        "name": "e2e-rollouts",
        "suites": ["ci-tiny"],
        "policy": {
            "kind": "mock",
            "policy_id": "mock-e2e",
            "seed": 2,
            "profile": {
                "correct_probability": 0.6,
                "timeout_probability": 0.1,
                "error_probability": 0.1,
            },
        },
        "rollouts": {
            "samples_per_task": 2,
            "concurrency": 4,
            "retry": {"max_attempts": 3, "backoff_initial_s": 0.0},
        },
        "sandbox": {"kind": "subprocess", "limits": {"wall_time_s": 5}},
    }
    config = tmp_path / "job.yaml"
    config.write_text(yaml.safe_dump(job), encoding="utf-8")
    results = tmp_path / "results"
    common = ["--config-dir", str(config_dir), "--results-dir", str(results)]

    first = RUNNER.invoke(app, [*common, "rollout", "run", str(config), "--run-id", "roll"])
    assert first.exit_code == 0, first.stdout
    summary = json.loads((results / "roll" / "rollout_summary.json").read_text())
    total = summary["completed"] + summary["failed"]
    assert total == summary["total_jobs"] > 0 and summary["retries"] > 0

    inspect = RUNNER.invoke(app, [*common, "trajectories", "inspect", "roll"])
    assert inspect.exit_code == 0 and "arithmetic" in inspect.stdout

    query = RUNNER.invoke(
        app, [*common, "trajectories", "query", "roll", "--domain", "python", "--json"]
    )
    assert query.exit_code == 0
    rows = json.loads(query.stdout)
    assert rows and {r["domain"] for r in rows} == {"python"}

    show = RUNNER.invoke(app, [*common, "trajectories", "show", "roll", rows[0]["trajectory_id"]])
    assert show.exit_code == 0 and "verification" in show.stdout

    resume = RUNNER.invoke(app, [*common, "rollout", "resume", "roll"])
    assert resume.exit_code == 0, resume.stdout
    resumed = json.loads((results / "roll" / "rollout_summary.json").read_text())
    assert resumed["skipped_existing"] == summary["completed"]
    assert resumed["attempted"] == summary["failed"]
