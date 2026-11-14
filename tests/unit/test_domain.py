from __future__ import annotations

import pytest
from pydantic import ValidationError

from rlfactory.domain.common import ComponentRef, canonical_json, stable_hash, stable_seed
from rlfactory.domain.model import GenerationConfig, ModelReference
from rlfactory.domain.result import EvaluationResult, MetricsSummary
from rlfactory.domain.reward import RewardBreakdown, RewardResult
from rlfactory.domain.task import (
    ArithmeticPayload,
    PythonFunctionPayload,
    PythonTestCase,
    StructuredOutputPayload,
    Task,
    TaskDomain,
    TaskMetadata,
)
from rlfactory.domain.training import (
    DatasetReference,
    DPOParams,
    SFTParams,
    TrainingConfig,
    TrainingMethod,
    TrainingRun,
)
from rlfactory.domain.trajectory import FailureKind, FailureOrigin, TrajectoryStatus
from rlfactory.domain.verification import VerificationResult
from tests.helpers import make_task, play


def test_stable_hash_is_deterministic_and_order_sensitive() -> None:
    assert stable_hash({"b": 1, "a": 2}) == stable_hash({"a": 2, "b": 1})
    assert stable_hash("a", "b") != stable_hash("b", "a")
    assert len(stable_hash("x", length=8)) == 8
    assert stable_seed("x") == stable_seed("x")
    with pytest.raises(ValueError, match="length"):
        stable_hash("x", length=2)


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": float("nan")})


def test_task_payload_must_match_domain() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        Task(
            task_id="t",
            domain=TaskDomain.PYTHON,
            prompt="p",
            environment="arithmetic",
            verifier="numeric",
            payload=ArithmeticPayload(expression="1+1", answer="2"),
            metadata=TaskMetadata(family="arithmetic.x"),
        )


def test_structured_payload_requires_ground_truth() -> None:
    with pytest.raises(ValidationError, match="expected value or constraints"):
        StructuredOutputPayload(json_schema={"type": "object"})
    with pytest.raises(ValidationError, match="reference answer"):
        StructuredOutputPayload(json_schema={"type": "object"}, constraints=("x > 1",))


def test_python_payload_validates_entry_point() -> None:
    with pytest.raises(ValidationError, match="identifier"):
        PythonFunctionPayload(
            entry_point="not valid",
            signature="def f()",
            tests=(PythonTestCase(args=(), expected=1),),
            reference_solution="def f(): return 1",
        )


def test_task_fingerprint_ignores_id_but_tracks_content() -> None:
    task = make_task("arithmetic.integer_chain", seed=1)
    renamed = task.model_copy(update={"task_id": "other"})
    assert task.fingerprint() == renamed.fingerprint()
    changed = task.model_copy(update={"prompt": task.prompt + " "})
    assert changed.fingerprint() != task.fingerprint()


def test_verification_result_bounds() -> None:
    ref = ComponentRef(name="v", version="1")
    with pytest.raises(ValidationError):
        VerificationResult(passed=False, score=1.5, reason_code="x", verifier=ref)
    with pytest.raises(ValidationError, match="positive score"):
        VerificationResult(passed=True, score=0.0, reason_code="ok", verifier=ref)


def test_reward_breakdown_sign_conventions_and_total() -> None:
    parts = RewardBreakdown(correctness=1.0, format=0.1, length=-0.2, tool_cost=-0.05)
    assert parts.total() == pytest.approx(0.85)
    assert parts.as_dict()["length"] == -0.2
    with pytest.raises(ValidationError):
        RewardBreakdown(length=0.5)
    with pytest.raises(ValidationError, match="finite"):
        RewardResult(
            total=float("inf"), components=parts, reward_fn=ComponentRef(name="r", version="1")
        )


def test_failure_origin_separates_model_and_infrastructure() -> None:
    assert FailureKind.INVALID_OUTPUT.origin is FailureOrigin.MODEL
    assert FailureKind.SANDBOX_TIMEOUT.origin is FailureOrigin.MODEL
    for kind in (
        FailureKind.POLICY_TIMEOUT,
        FailureKind.POLICY_ERROR,
        FailureKind.SANDBOX_ERROR,
        FailureKind.VERIFIER_ERROR,
        FailureKind.INFRASTRUCTURE_ERROR,
    ):
        assert kind.origin is FailureOrigin.INFRASTRUCTURE


def test_trajectory_invariants_and_finalize() -> None:
    task = make_task("arithmetic.percentage", seed=2)
    trajectory = play(task, [f"<answer>{task.payload.answer}</answer>"])  # type: ignore[union-attr]
    assert trajectory.status is TrajectoryStatus.COMPLETED
    assert trajectory.passed and trajectory.is_trainable
    assert trajectory.prompt_messages()[-1].role.value == "user"
    with pytest.raises(ValueError, match="pending"):
        trajectory.finalized(trajectory.verification, trajectory.rewards)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="verified"):
        type(trajectory).model_validate({**trajectory.model_dump(), "verification": None})


def test_generation_config_and_model_reference() -> None:
    assert GenerationConfig(temperature=0).is_greedy
    assert GenerationConfig().fingerprint() == GenerationConfig().fingerprint()
    assert not ModelReference(name="m").is_pinned
    assert ModelReference(name="m", revision="0123456789abcdef").is_pinned
    assert "unpinned" in ModelReference(name="m").describe()


def test_training_run_method_must_match_params() -> None:
    dataset = DatasetReference(name="d", path="p", schema_name="s", num_records=1, fingerprint="f")
    with pytest.raises(ValidationError, match="does not match"):
        TrainingRun(
            run_id="r",
            method=TrainingMethod.DPO,
            model=ModelReference(name="m"),
            dataset=dataset,
            config=TrainingConfig(method_params=SFTParams()),
        )
    run = TrainingRun(
        run_id="r",
        method=TrainingMethod.DPO,
        model=ModelReference(name="m"),
        dataset=dataset,
        config=TrainingConfig(method_params=DPOParams()),
    )
    assert run.fingerprint() == run.model_copy(update={"run_id": "other"}).fingerprint()


def test_metrics_summary_metric_lookup() -> None:
    summary = MetricsSummary(
        num_tasks=1,
        num_trajectories=2,
        num_completed=2,
        num_infra_failures=0,
        verified_accuracy=0.5,
        pass_at_k={4: 0.9},
    )
    assert summary.metric("verified_accuracy") == 0.5
    assert summary.metric("pass_at_4") == 0.9
    assert summary.metric("format_compliance") is None
    with pytest.raises(KeyError):
        summary.metric("failure_categories")


def test_evaluation_result_scope_parsing() -> None:
    summary = MetricsSummary(num_tasks=0, num_trajectories=0, num_completed=0, num_infra_failures=0)
    from datetime import UTC, datetime

    from rlfactory.domain.common import CodeVersion

    result = EvaluationResult(
        evaluation_id="e",
        label="l",
        policy_id="p",
        suite="s",
        num_tasks=0,
        samples_per_task=1,
        temperature=0.0,
        overall=summary,
        by_domain={"python": summary},
        created_at=datetime.now(UTC),
        code=CodeVersion(rlfactory_version="t"),
    )
    assert result.scope("overall") is summary
    assert result.scope("domain:python") is summary
    assert result.scope("domain:missing") is None
    with pytest.raises(ValueError, match="scope"):
        result.scope("bogus")
