"""Render ``report.md`` (and figures) for an experiment run from its stored artefacts only.

The report never computes new results: it formats what the stages measured. Missing artefacts
render as "not run". Simulated runs carry a prominent banner and per-row labels.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlfactory.config.loader import read_json_model
from rlfactory.domain.experiment import RunManifest
from rlfactory.domain.result import EvaluationResult, GateReport, MetricsSummary
from rlfactory.domain.training import TrainingSummary
from rlfactory.reporting.plots import (
    Series,
    grouped_bar_chart,
    histogram_chart,
    horizontal_bar_chart,
    line_chart,
    write_svg,
)

SIMULATED_BANNER = (
    "> **SIMULATED RESULTS.** This run used the mock policy and/or the mock training backend. "
    "Every number below characterises the RLFactory infrastructure on synthetic tasks; none of it "
    "is a measurement of a real model, and no claim about SFT, DPO or GRPO effectiveness follows."
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def ci(summary: MetricsSummary) -> str:
    if summary.accuracy_ci95 is None:
        return "n/a"
    low, high = summary.accuracy_ci95
    return f"{low * 100:.1f}-{high * 100:.1f}%"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _load_eval(run_dir: Path, label: str) -> EvaluationResult | None:
    path = run_dir / "evaluation" / f"{label}.json"
    return read_json_model(path, EvaluationResult) if path.exists() else None


def _section_experiment(manifest: RunManifest | None, config: dict[str, Any] | None) -> list[str]:
    lines = ["## Experiment", ""]
    if manifest is None:
        return [*lines, "Manifest not found.", ""]
    code = manifest.code
    dirty = " (uncommitted changes)" if code.git_dirty else ""
    lines += _table(
        ["Field", "Value"],
        [
            ["Run ID", f"`{manifest.run_id}`"],
            ["Experiment", manifest.experiment],
            ["Created (UTC)", manifest.created_at.strftime("%Y-%m-%d %H:%M:%S")],
            [
                "RLFactory",
                f"{code.rlfactory_version}, git {code.git_sha[:12] if code.git_sha else 'unknown'}{dirty}",
            ],
            ["Seed", str(manifest.seed)],
            ["Config fingerprint", f"`{manifest.config_fingerprint}`"],
            ["Simulated", "yes" if manifest.simulated else "no"],
        ],
    )
    if config and config.get("description"):
        lines += ["", str(config["description"])]
    return [*lines, ""]


def _section_tasks(manifest: RunManifest | None, summary: dict[str, Any] | None) -> list[str]:
    lines = ["## Tasks", ""]
    if manifest is not None:
        lines += _table(
            ["Suite", "Version", "Tasks", "Fingerprint"],
            [
                [s.name, s.version, str(s.num_tasks), f"`{s.fingerprint}`"]
                for s in manifest.task_suites
            ],
        )
        lines.append("")
    if summary:
        lines.append(
            f"After validation and decontamination: **{summary['train']}** training, "
            f"**{summary['validation']}** development and **{summary['evaluation']}** held-out "
            f"evaluation tasks; {summary['invalid']} generated tasks failed validation."
        )
        removed = summary.get("contamination_removed") or {}
        if removed:
            lines.append(
                f"Contamination filter removed: {', '.join(f'{k}={v}' for k, v in removed.items())}."
            )
        domains = sorted(
            set(summary.get("train_by_domain", {})) | set(summary.get("evaluation_by_domain", {}))
        )
        lines += [
            "",
            *_table(
                ["Domain", "Train", "Eval"],
                [
                    [
                        d,
                        str(summary["train_by_domain"].get(d, 0)),
                        str(summary["evaluation_by_domain"].get(d, 0)),
                    ]
                    for d in domains
                ],
            ),
        ]
        coverage = summary.get("verifier_coverage") or {}
        if coverage:
            total = sum(coverage.values()) or 1
            lines += [
                "",
                "Verifier coverage: "
                + ", ".join(
                    f"{kind} {count} ({count / total:.0%})"
                    for kind, count in sorted(coverage.items())
                ),
            ]
    return [*lines, ""]


def _section_environments(manifest: RunManifest | None, config: dict[str, Any] | None) -> list[str]:
    lines = ["## Environments and verifiers", ""]
    if manifest is not None:
        lines += _table(
            ["Component", "Name", "Version"],
            [["environment", e.name, e.version] for e in manifest.environments]
            + [["verifier", v.name, v.version] for v in manifest.verifiers]
            + [["reward", r.name, r.version] for r in manifest.rewards],
        )
    if config:
        sandbox = config.get("sandbox", {}).get("kind", "subprocess")
        lines += ["", f"Sandbox: `{sandbox}` (see docs/security/sandbox.md for its guarantees)."]
    return [*lines, ""]


def _section_policy(manifest: RunManifest | None) -> list[str]:
    if manifest is None:
        return []
    policy, generation = manifest.policy, manifest.generation
    lines = [
        "## Policy",
        "",
        f"Base policy `{policy.policy_id}` ({policy.kind}, model `{policy.model}`).",
    ]
    if generation is not None:
        lines.append(
            f"Generation: temperature {generation.temperature}, top_p {generation.top_p}, "
            f"max_tokens {generation.max_tokens}."
        )
    return [*lines, ""]


def _section_rollouts(rollouts: dict[str, Any] | None) -> list[str]:
    lines = ["## Rollouts", ""]
    if rollouts is None:
        return [*lines, "Not run.", ""]
    tokens = f"{rollouts['completion_tokens']}" + (
        " (estimated)" if rollouts.get("tokens_estimated") else ""
    )
    lines += _table(
        ["Metric", "Value"],
        [
            [
                "Trajectories (completed / infra-failed)",
                f"{rollouts['completed']} / {rollouts['failed']}",
            ],
            ["Skipped (already completed, resume)", str(rollouts["skipped_existing"])],
            ["Verified pass rate", pct(rollouts.get("pass_rate"))],
            ["Mean reward", num(rollouts.get("mean_reward"))],
            ["Retries (infrastructure only)", str(rollouts["retries"])],
            ["Infrastructure failure rate", pct(rollouts.get("infra_failure_rate"))],
            ["Verification rate", pct(rollouts.get("verification_rate"))],
            ["Throughput", f"{num(rollouts.get('trajectories_per_s'), 1)} trajectories/s"],
            ["Completion tokens", tokens],
            ["Wall time", f"{num(rollouts.get('wall_time_s'), 1)} s"],
        ],
    )
    if rollouts.get("failures_by_kind"):
        lines += [
            "",
            "Failures by kind: "
            + ", ".join(f"`{k}` {v}" for k, v in rollouts["failures_by_kind"].items()),
        ]
    by_domain = rollouts.get("by_domain") or {}
    if by_domain:
        lines += [
            "",
            *_table(
                ["Domain", "Completed", "Failed", "Pass rate", "Mean reward"],
                [
                    [
                        d,
                        f"{v['completed']:.0f}",
                        f"{v['failed']:.0f}",
                        pct(v["pass_rate"]),
                        num(v["mean_reward"]),
                    ]
                    for d, v in by_domain.items()
                ],
            ),
        ]
    return [*lines, ""]


def _section_reward(analysis: dict[str, Any] | None) -> list[str]:
    lines = ["## Reward", ""]
    if analysis is None:
        return [*lines, "Analysis stage not run.", ""]
    calibration = analysis.get("calibration", {})
    lines += _table(
        ["Reward calibration", "Value"],
        [
            [
                "Mean reward (passed / failed)",
                f"{num(calibration.get('mean_reward_passed'))} / {num(calibration.get('mean_reward_failed'))}",
            ],
            ["Point-biserial r(reward, verified)", num(calibration.get("point_biserial"))],
            ["AUC P(reward_pass > reward_fail)", num(calibration.get("auc"))],
            ["Reward std", num(calibration.get("reward_std"))],
        ],
    )
    components = calibration.get("component_means") or {}
    if components:
        lines += [
            "",
            "Mean reward components: " + ", ".join(f"{k} {v:+.3f}" for k, v in components.items()),
        ]
    hacking = analysis.get("hacking", {})
    counts = hacking.get("counts") or {}
    lines += [
        "",
        (
            f"Reward-hacking heuristics flagged: {', '.join(f'{k} {v}' for k, v in counts.items()) or 'nothing'} "
            f"(of {hacking.get('examined', 0)} trajectories). {hacking.get('caveat', '')}"
        ),
    ]
    return [*lines, ""]


def _section_datasets(run_dir: Path) -> list[str]:
    lines = ["## Datasets", ""]
    rows = []
    for card_path in sorted((run_dir / "datasets").glob("*/dataset_card.json")):
        card = _json(card_path)
        excluded = card.get("build_report", {}).get("excluded", {})
        rows.append(
            [
                card["kind"],
                f"`{card['schema_name']}`",
                str(card["num_records"]),
                ", ".join(f"{k}={v}" for k, v in excluded.items()) or "-",
                "ok" if card.get("validation", {}).get("ok") else "FAILED",
            ]
        )
    if not rows:
        return [*lines, "No datasets built.", ""]
    return [*lines, *_table(["Kind", "Schema", "Records", "Excluded", "Validation"], rows), ""]


def _section_training(training: TrainingSummary | None) -> list[str]:
    lines = ["## Training", ""]
    if training is None:
        return [*lines, "Not run.", ""]
    simulated = " (simulated: no weights trained)" if training.simulated else ""
    lines += _table(
        ["Field", "Value"],
        [
            ["Backend / method", f"{training.backend} / {training.method.value}{simulated}"],
            ["Base model", training.model.describe()],
            ["Dataset", f"{training.dataset.name} ({training.dataset.num_records} records)"],
            ["State / steps", f"{training.state.value} / {training.steps}"],
            ["Final loss", num(training.final_loss, 4)],
            ["Wall time", f"{training.wall_time_s:.1f} s"],
            ["Tokens seen", str(training.tokens_seen) if training.tokens_seen else "n/a"],
            ["Peak VRAM", f"{training.peak_vram_gb:.2f} GB" if training.peak_vram_gb else "n/a"],
            ["Framework", training.framework_version or "n/a"],
            [
                "Resumed from step",
                str(training.resumed_from_step) if training.resumed_from_step else "-",
            ],
        ],
    )
    return [*lines, ""]


def _python_pass(result: EvaluationResult) -> str:
    python = result.by_domain.get("python")
    return pct(python.verified_accuracy) if python else "n/a"


def _section_evaluation(
    base: EvaluationResult | None,
    candidate: EvaluationResult | None,
    training: TrainingSummary | None,
) -> list[str]:
    lines = ["## Evaluation", ""]
    results = [r for r in (base, candidate) if r is not None]
    if not results:
        return [*lines, "Not run.", ""]
    rows = []
    for result in results:
        is_candidate = result is candidate
        tag = " (simulated)" if result.simulated else ""
        method = training.method.value if is_candidate and training else "base"
        train_time = (
            ("simulated" if training.simulated else f"{training.wall_time_s / 60:.1f} min")
            if is_candidate and training
            else "-"
        )
        vram = (
            f"{training.peak_vram_gb:.1f} GB"
            if is_candidate and training and training.peak_vram_gb
            else "-"
        )
        rows.append(
            [
                f"`{result.policy_id}`{tag}",
                method,
                pct(result.overall.verified_accuracy),
                _python_pass(result),
                pct(result.overall.format_compliance),
                num(result.overall.mean_reward),
                train_time,
                vram,
            ]
        )
    lines += _table(
        [
            "Model",
            "Method",
            "Verified Acc.",
            "Python Pass",
            "Format",
            "Reward",
            "Train Time",
            "VRAM",
        ],
        rows,
    )
    first = results[0]
    lines += [
        "",
        (
            f"Suite `{first.suite}`: {first.num_tasks} tasks x {first.samples_per_task} samples, "
            f"temperature {first.temperature}. Infrastructure failures are excluded from accuracy and "
            "reported separately."
        ),
        "",
    ]
    domains = sorted({d for r in results for d in r.by_domain})
    header = ["Domain"] + [f"{r.label} acc. (95% CI)" for r in results]
    domain_rows = [
        [d]
        + [
            f"{pct(r.by_domain[d].verified_accuracy)} ({ci(r.by_domain[d])})"
            if d in r.by_domain
            else "n/a"
            for r in results
        ]
        for d in domains
    ]
    domain_rows.append(
        ["**overall**"] + [f"{pct(r.overall.verified_accuracy)} ({ci(r.overall)})" for r in results]
    )
    lines += _table(header, domain_rows)
    extra = [
        f"{result.label}: "
        + ", ".join(f"pass@{k} {pct(v)}" for k, v in sorted(result.overall.pass_at_k.items()))
        for result in results
        if result.overall.pass_at_k
    ]
    if extra:
        lines += ["", "Unbiased pass@k (stochastic sampling): " + "; ".join(extra)]
    metric_rows = []
    for result in results:
        o = result.overall
        metric_rows.append(
            [
                result.label,
                pct(o.pass_at_1),
                pct(o.tool_success_rate),
                pct(o.sandbox_failure_rate),
                pct(o.infra_failure_rate),
                num(o.mean_completion_tokens, 1),
                num(o.latency_p50_s, 3),
                num(o.reward_correctness_correlation),
            ]
        )
    lines += [
        "",
        *_table(
            [
                "Policy",
                "pass@1",
                "Tool success",
                "Sandbox failures",
                "Infra failures",
                "Mean tokens",
                "p50 latency (s)",
                "r(reward, verified)",
            ],
            metric_rows,
        ),
    ]
    return [*lines, ""]


def _section_gate(gate: GateReport | None, promotion: dict[str, Any] | None) -> list[str]:
    lines = ["## Regression gate", ""]
    if gate is None:
        return [*lines, "Not run.", ""]
    lines += _table(
        ["Rule", "Scope", "Baseline", "Candidate", "Result"],
        [
            [
                c.rule.name,
                c.rule.scope,
                num(c.baseline, 4),
                num(c.candidate, 4),
                "PASS" if c.passed else "FAIL",
            ]
            for c in gate.checks
        ],
    )
    lines += [
        "",
        f"**Gate: {'PASSED' if gate.passed else 'FAILED'}** ({gate.candidate_label} vs {gate.baseline_label}).",
    ]
    lines += [f"- {warning}" for warning in gate.warnings]
    if promotion:
        lines.append(
            f"\nPromotion decision for `{promotion['checkpoint_id']}`: **{promotion['state']}**."
        )
    return [*lines, ""]


def _section_failures(
    candidate: EvaluationResult | None, analysis: dict[str, Any] | None
) -> list[str]:
    lines = ["## Failure analysis", ""]
    rows = []
    if candidate is not None:
        rows += [
            [f"evaluation ({candidate.label})", k, str(v)]
            for k, v in sorted(candidate.overall.failure_categories.items())
        ]
    if analysis is not None:
        rows += [
            ["training rollouts", k, str(v)]
            for k, v in sorted(analysis.get("failures", {}).get("counts", {}).items())
        ]
    if not rows:
        return [*lines, "No failure data.", ""]
    lines += _table(["Source", "Observed category", "Count"], rows)
    lines += [
        "",
        (
            "Categories describe what was observed (format, runtime error, timeout, wrong answer, tool "
            "misuse, verifier or infrastructure failure). Per-trajectory speculative root causes are kept "
            "in a separate, explicitly labelled `hypothesis` column of `failures.parquet`."
        ),
    ]
    return [*lines, ""]


def _section_flywheel(flywheel: list[dict[str, Any]] | None) -> list[str]:
    if not flywheel:
        return []
    lines = ["## Failure-driven data flywheel", ""]
    lines += _table(
        ["Iteration", "Failure Tasks Added", "Train Examples", "Eval Accuracy", "Gate"],
        [
            [
                str(i["iteration"]),
                str(i["tasks_added"]),
                str(i["train_examples"]),
                pct(i["eval_accuracy"]) + (" (simulated)" if i.get("simulated") else ""),
                "-" if i["gate_passed"] is None else ("pass" if i["gate_passed"] else "fail"),
            ]
            for i in flywheel
        ],
    )
    return [*lines, ""]


def _section_resources(run_dir: Path, manifest: RunManifest | None) -> list[str]:
    lines = ["## Resource usage", ""]
    rows = []
    for path in sorted((run_dir / "stages").glob("*.json")):
        record = _json(path)
        res = record.get("resources", {})
        rows.append(
            [
                record["name"],
                record["status"] + (" (cached)" if record.get("cached") else ""),
                f"{record['duration_s']:.1f}",
                num(res.get("peak_rss_gb"), 2),
                num(res.get("mean_cpu_percent"), 0),
            ]
        )
    if rows:
        lines += _table(["Stage", "Status", "Wall (s)", "Peak RSS (GB)", "Mean CPU %"], rows)
    if manifest is not None:
        hw = manifest.hardware
        lines += [
            "",
            f"Hardware: {hw.platform}, Python {hw.python}, {hw.cpu_count} CPUs, {hw.ram_gb:.1f} GB RAM, GPU: {hw.gpu or 'none'}.",
        ]
    return [*lines, ""]


def _limitations(
    manifest: RunManifest | None,
    rollouts: dict[str, Any] | None,
    config: dict[str, Any] | None,
    candidate: EvaluationResult | None,
) -> list[str]:
    items = []
    if manifest is not None and manifest.simulated:
        items.append(
            "Model behaviour is simulated (mock policy / mock training); results describe the pipeline, not a model."
        )
    if rollouts and rollouts.get("tokens_estimated"):
        items.append("Token counts are whitespace approximations reported by the mock policy.")
    if config and config.get("sandbox", {}).get("kind", "subprocess") == "subprocess":
        items.append(
            "Generated Python ran in the subprocess sandbox, which is not a security boundary."
        )
    if candidate is not None and candidate.overall.num_completed < 200:
        items.append(
            "Evaluation samples are small; read differences against the Wilson 95% intervals."
        )
    items.append(
        "Task difficulty levels are heuristic signals, not measures of cognitive difficulty."
    )
    items.append(
        "Reward-hacking detectors are heuristics; an empty report does not prove absence of hacking."
    )
    return ["## Limitations", "", *[f"- {item}" for item in items], ""]


def _figures(
    run_dir: Path,
    base: EvaluationResult | None,
    candidate: EvaluationResult | None,
    analysis: dict[str, Any] | None,
    flywheel: list[dict[str, Any]] | None,
) -> list[str]:
    plots = run_dir / "plots"
    made: list[tuple[str, str]] = []
    results = [r for r in (base, candidate) if r is not None]
    if results:
        domains = sorted({d for r in results for d in r.by_domain})
        series = [
            Series(
                r.label,
                [r.by_domain[d].verified_accuracy if d in r.by_domain else None for d in domains],
            )
            for r in results
        ]
        suffix = " (simulated)" if any(r.simulated for r in results) else ""
        write_svg(
            plots / "accuracy_by_domain.svg",
            grouped_bar_chart(
                f"Verified accuracy by domain{suffix}",
                domains,
                series,
                subtitle="Held-out evaluation suite; infrastructure failures excluded",
            ),
        )
        made.append(("Verified accuracy by domain", "plots/accuracy_by_domain.svg"))
    histogram = (analysis or {}).get("calibration", {}).get("histogram") or []
    if histogram:
        write_svg(
            plots / "reward_distribution.svg",
            histogram_chart(
                "Reward distribution (training rollouts)",
                [tuple(b) for b in histogram],
                subtitle="Scalar reward per completed trajectory",
                x_label="reward",
            ),
        )
        made.append(("Reward distribution", "plots/reward_distribution.svg"))
    if candidate is not None and candidate.overall.failure_categories:
        write_svg(
            plots / "failure_categories.svg",
            horizontal_bar_chart(
                "Observed failure categories (candidate evaluation)",
                sorted(candidate.overall.failure_categories.items()),
                subtitle="Counts of failed trajectories by observed category",
            ),
        )
        made.append(("Failure categories", "plots/failure_categories.svg"))
    if flywheel:
        write_svg(
            plots / "flywheel_accuracy.svg",
            line_chart(
                "Held-out accuracy by flywheel iteration"
                + (" (simulated)" if flywheel[-1].get("simulated") else ""),
                [str(i["iteration"]) for i in flywheel],
                [i["eval_accuracy"] for i in flywheel],
                subtitle="Iteration 0 is the base policy",
            ),
        )
        made.append(("Flywheel accuracy", "plots/flywheel_accuracy.svg"))
    if not made:
        return []
    return ["## Figures", "", *[f"![{title}]({path})" for title, path in made], ""]


def generate_report(run_dir: Path) -> Path:
    manifest = (
        read_json_model(run_dir / "manifest.json", RunManifest)
        if (run_dir / "manifest.json").exists()
        else None
    )
    config_path = run_dir / "config.yaml"
    config: dict[str, Any] | None = None
    if config_path.exists():
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rollouts = _json(run_dir / "rollout_summary.json")
    analysis = _json(run_dir / "reward_analysis.json")
    training = (
        read_json_model(run_dir / "training_summary.json", TrainingSummary)
        if (run_dir / "training_summary.json").exists()
        else None
    )
    base, candidate = _load_eval(run_dir, "base"), _load_eval(run_dir, "candidate")
    gate = (
        read_json_model(run_dir / "gate_report.json", GateReport)
        if (run_dir / "gate_report.json").exists()
        else None
    )
    evaluation_bundle = _json(run_dir / "evaluation.json") or {}
    flywheel = _json(run_dir / "flywheel.json")
    title = manifest.experiment if manifest else run_dir.name
    lines = [f"# RLFactory report: {title}", ""]
    if manifest is None or manifest.simulated:
        lines += [SIMULATED_BANNER, ""]
    lines += _section_experiment(manifest, config)
    lines += _section_tasks(manifest, _json(run_dir / "tasks" / "summary.json"))
    lines += _section_environments(manifest, config)
    lines += _section_policy(manifest)
    lines += _section_rollouts(rollouts)
    lines += _section_reward(analysis)
    lines += _section_datasets(run_dir)
    lines += _section_training(training)
    lines += _section_evaluation(base, candidate, training)
    lines += _section_gate(gate, evaluation_bundle.get("promotion"))
    lines += _section_failures(candidate, analysis)
    lines += _section_flywheel(flywheel)
    lines += _figures(run_dir, base, candidate, analysis, flywheel)
    lines += _section_resources(run_dir, manifest)
    lines += _limitations(manifest, rollouts, config, candidate)
    lines += [
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by RLFactory from stored run artefacts._",
        "",
    ]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
