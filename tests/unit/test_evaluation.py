from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from rlfactory.domain.common import CodeVersion
from rlfactory.domain.result import (
    EvaluationResult,
    FailureCategory,
    GateRule,
    GateRuleKind,
    MetricsSummary,
    PromotionState,
)
from rlfactory.domain.task import TaskDomain
from rlfactory.domain.trajectory import FailureKind, FailureRecord, TrajectoryStatus
from rlfactory.evaluation.coverage import verifier_coverage
from rlfactory.evaluation.evaluator import EvaluationConfig, Evaluator
from rlfactory.evaluation.external import (
    LMEvalHarnessConfig,
    build_lm_eval_command,
    parse_lm_eval_results,
)
from rlfactory.evaluation.failures import analyze_failures, classify, write_failures_parquet
from rlfactory.evaluation.gates import DEFAULT_GATES, GateConfig, evaluate_gates
from rlfactory.evaluation.metrics import pass_at_k, summarize_trajectories, wilson_interval
from rlfactory.evaluation.promotion import PromotionError, PromotionRegistry
from rlfactory.policies.mock import MockPolicy, MockPolicyProfile
from rlfactory.policies.oracle import ResponseMode
from rlfactory.verifiers.registry import RegexSpec, VerifierRegistry
from tests.helpers import deps, make_task, oracle_play, play, tasks_for


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (5, 0, 1, 0.0),
        (5, 5, 3, 1.0),
        (4, 1, 1, 0.25),
        (4, 1, 2, 0.5),
        (10, 3, 5, 1 - math.comb(7, 5) / math.comb(10, 5)),
    ],
)
def test_pass_at_k_unbiased_estimator(n: int, c: int, k: int, expected: float) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_pass_at_k_input_validation() -> None:
    with pytest.raises(ValueError, match="invalid pass@k"):
        pass_at_k(2, 3, 1)
    with pytest.raises(ValueError, match="invalid pass@k"):
        pass_at_k(2, 1, 3)


def test_wilson_interval() -> None:
    low, high = wilson_interval(5, 10)  # type: ignore[misc]
    assert 0.2 < low < 0.5 < high < 0.8
    assert wilson_interval(0, 0) is None
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)  # type: ignore[index]


def _mixed_trajectories():  # type: ignore[no-untyped-def]
    task = make_task("arithmetic.integer_chain", seed=31)
    right = f"<answer>{task.payload.answer}</answer>"  # type: ignore[union-attr]
    completed = [
        play(task, [right], sample_index=0),
        play(task, [right], sample_index=1),
        play(task, ["<answer>-1</answer>"], sample_index=2),
        play(task, ["no format"], sample_index=3),
    ]
    failed = completed[0].model_copy(
        update={
            "trajectory_id": "infra",
            "status": TrajectoryStatus.FAILED,
            "verification": None,
            "rewards": None,
            "failure": FailureRecord(kind=FailureKind.POLICY_TIMEOUT),
        }
    )
    return [*completed, failed]


def test_summary_metrics_exclude_infrastructure_failures() -> None:
    summary = summarize_trajectories(_mixed_trajectories(), ks=(1, 2, 4), temperature=0.7)
    assert (
        summary.num_trajectories == 5
        and summary.num_completed == 4
        and summary.num_infra_failures == 1
    )
    assert summary.verified_accuracy == 0.5 and summary.pass_at_1 == 0.5
    assert summary.pass_at_k[2] == pytest.approx(1 - math.comb(2, 2) / math.comb(4, 2))
    assert summary.format_compliance == 0.75 and summary.infra_failure_rate == 0.2
    assert summary.failure_categories == {
        "reasoning_error": 1,
        "format_error": 1,
        "infrastructure_failure": 1,
    }
    greedy = summarize_trajectories(_mixed_trajectories(), ks=(1, 2), temperature=0.0)
    assert greedy.pass_at_k == {}


def test_failure_classification_keeps_hypotheses_separate(tmp_path: Path) -> None:
    task = make_task("arithmetic.integer_chain", seed=33)
    value = int(task.payload.answer)  # type: ignore[union-attr]
    near_miss = classify(play(task, [f"<answer>{value + 1}</answer>"]))
    assert near_miss is not None and near_miss.category is FailureCategory.REASONING_ERROR
    assert near_miss.hypothesis is not None and near_miss.hypothesis.startswith("SPECULATIVE")
    unformatted = classify(play(task, [f"The answer is {value}."]))
    assert unformatted is not None and unformatted.category is FailureCategory.FORMAT_ERROR
    assert classify(play(task, [f"<answer>{value}</answer>"])) is None
    tool_task = make_task("tool_use.inventory_value", seed=2)
    misuse = classify(oracle_play(tool_task, ResponseMode.MALFORMED))
    assert misuse is not None and misuse.category in {
        FailureCategory.TOOL_MISUSE,
        FailureCategory.FORMAT_ERROR,
    }
    analysis = analyze_failures(_mixed_trajectories())
    assert analysis.failed_task_ids() == [
        task.task_id for task in [make_task("arithmetic.integer_chain", seed=31)]
    ]
    write_failures_parquet(analysis, tmp_path / "failures.parquet")
    table = pq.read_table(tmp_path / "failures.parquet")
    assert {"observed", "hypothesis", "category"} <= set(table.column_names)


def _result(
    label: str, accuracy: float, python: float, *, simulated: bool = False
) -> EvaluationResult:
    overall = MetricsSummary(
        num_tasks=10,
        num_trajectories=20,
        num_completed=20,
        num_infra_failures=0,
        verified_accuracy=accuracy,
        accuracy_ci95=wilson_interval(int(accuracy * 20), 20),
        format_compliance=0.95,
        infra_failure_rate=0.0,
    )
    domain = overall.model_copy(update={"verified_accuracy": python})
    return EvaluationResult(
        evaluation_id=label,
        label=label,
        policy_id=label,
        suite="s",
        num_tasks=10,
        samples_per_task=2,
        temperature=0.0,
        overall=overall,
        by_domain={"python": domain},
        simulated=simulated,
        created_at=datetime.now(UTC),
        code=CodeVersion(rlfactory_version="t"),
    )


def test_gates_regressions_thresholds_and_warnings() -> None:
    baseline = _result("base", 0.60, 0.50)
    assert evaluate_gates(baseline, _result("cand", 0.59, 0.50), DEFAULT_GATES).passed
    regressed = evaluate_gates(baseline, _result("cand", 0.55, 0.50), DEFAULT_GATES)
    assert not regressed.passed and not regressed.checks[0].passed
    python_drop = evaluate_gates(baseline, _result("cand", 0.70, 0.45), DEFAULT_GATES)
    assert not python_drop.passed and [c.passed for c in python_drop.checks][1] is False
    assert any("noise" in w for w in python_drop.warnings)
    simulated = evaluate_gates(baseline, _result("cand", 0.6, 0.5, simulated=True), DEFAULT_GATES)
    assert any("simulated" in w for w in simulated.warnings)
    missing_scope = GateConfig(
        rules=(
            GateRule(
                name="tools",
                metric="verified_accuracy",
                scope="domain:tool_use",
                kind=GateRuleKind.NO_REGRESSION,
            ),
        )
    )
    assert not evaluate_gates(baseline, _result("c", 0.6, 0.5), missing_scope).passed
    optional = GateConfig(
        rules=(
            GateRule(
                name="opt",
                metric="verified_accuracy",
                kind=GateRuleKind.MIN_VALUE,
                threshold=0.99,
                required=False,
            ),
        )
    )
    assert evaluate_gates(baseline, _result("c", 0.6, 0.5), optional).passed


def test_promotion_state_machine(tmp_path: Path) -> None:
    registry = PromotionRegistry(tmp_path / "registry.json")
    registry.register_candidate("ckpt-1", simulated=True)
    report = evaluate_gates(_result("b", 0.6, 0.5), _result("c", 0.7, 0.6), DEFAULT_GATES)
    decided = registry.decide("ckpt-1", report, evaluation_id="e1")
    assert decided.state is PromotionState.VALIDATED and decided.simulated
    with pytest.raises(PromotionError, match="already"):
        registry.decide("ckpt-1", report)
    with pytest.raises(PromotionError, match="never registered"):
        registry.decide("ckpt-2", report)
    registry.register_candidate("ckpt-3")
    rejected = registry.decide(
        "ckpt-3", evaluate_gates(_result("b", 0.6, 0.5), _result("c", 0.1, 0.1), DEFAULT_GATES)
    )
    assert rejected.state is PromotionState.REJECTED and rejected.reasons
    assert {r.checkpoint_id for r in registry.all()} == {"ckpt-1", "ckpt-3"}


def test_verifier_coverage() -> None:
    tasks = tasks_for(TaskDomain.ARITHMETIC)
    registry = VerifierRegistry({"regexy": RegexSpec(pattern="x")})
    coverage = verifier_coverage(tasks, registry, {TaskDomain.ARITHMETIC: "regexy"})
    assert coverage.counts["heuristic"] == len(tasks) and coverage.fraction("deterministic") == 0.0
    missing = verifier_coverage(tasks, registry, {TaskDomain.ARITHMETIC: "does-not-exist"})
    assert missing.counts["unverifiable"] == len(tasks)


def test_evaluator_end_to_end_on_mock_policy(tmp_path: Path) -> None:
    tasks = tasks_for(TaskDomain.TRANSFORMATION)
    policy = MockPolicy(
        tasks, MockPolicyProfile(correct_probability=1.0, malformed_probability=0.0)
    )
    config = EvaluationConfig(samples_per_task=2, pass_at_k=(1, 2))
    result, trajectories, summary = asyncio.run(
        Evaluator(deps()).evaluate(
            label="base",
            suite="t",
            policy=policy,
            tasks=tasks,
            config=config,
            run_id="r",
            store_root=tmp_path / "store",
        )
    )
    assert result.overall.verified_accuracy == 1.0 and result.simulated
    assert (
        set(result.by_domain) == {"transformation"} and len(trajectories) == summary.completed == 12
    )
    assert result.overall.pass_at_k == {}  # greedy decoding: pass@k is not meaningful


def test_lm_eval_adapter(tmp_path: Path) -> None:
    command = build_lm_eval_command(
        LMEvalHarnessConfig(model="m", base_url="http://h:8000/", tasks=("gsm8k", "mmlu"), limit=5)
    )
    assert command[:3] == ["lm_eval", "--model", "local-chat-completions"]
    assert "model=m,base_url=http://h:8000/v1/chat/completions,num_concurrent=4" in command
    assert "--apply_chat_template" in command and command[-2:] == ["--limit", "5"]
    with pytest.raises(ValueError, match="base_url"):
        build_lm_eval_command(LMEvalHarnessConfig(model="m"))
    assert (
        build_lm_eval_command(LMEvalHarnessConfig(model="m", model_backend="hf"))[4]
        == "pretrained=m"
    )
    results = tmp_path / "results.json"
    results.write_text(
        '{"results": {"gsm8k": {"exact_match,strict-match": 0.4, "alias": "gsm8k", "flag": true}}}',
        encoding="utf-8",
    )
    assert parse_lm_eval_results(results) == {"gsm8k": {"exact_match,strict-match": 0.4}}
