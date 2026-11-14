"""Evaluation results, regression gates and promotion decisions."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from rlfactory.domain.common import CodeVersion, DomainModel


class FailureCategory(StrEnum):
    """Observed failure categories (what happened, not why)."""

    REASONING_ERROR = "reasoning_error"
    FORMAT_ERROR = "format_error"
    CODE_RUNTIME_ERROR = "code_runtime_error"
    TIMEOUT = "timeout"
    TOOL_MISUSE = "tool_misuse"
    VERIFIER_FAILURE = "verifier_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class MetricsSummary(DomainModel):
    num_tasks: int = Field(ge=0)
    num_trajectories: int = Field(ge=0)
    num_completed: int = Field(ge=0)
    num_infra_failures: int = Field(ge=0)
    verified_accuracy: float | None = None
    accuracy_ci95: tuple[float, float] | None = None
    pass_at_1: float | None = None
    pass_at_k: dict[int, float] = Field(default_factory=dict)
    format_compliance: float | None = None
    tool_success_rate: float | None = None
    sandbox_failure_rate: float | None = None
    infra_failure_rate: float | None = None
    mean_reward: float | None = None
    reward_std: float | None = None
    reward_correctness_correlation: float | None = None
    mean_completion_tokens: float | None = None
    mean_completion_chars: float | None = None
    total_completion_tokens: int = 0
    latency_p50_s: float | None = None
    latency_p95_s: float | None = None
    failure_categories: dict[str, int] = Field(default_factory=dict)

    def metric(self, name: str) -> float | None:
        """Look up a scalar metric by name (used by regression gates)."""
        if name.startswith("pass_at_") and name != "pass_at_1":
            return self.pass_at_k.get(int(name.removeprefix("pass_at_")))
        value = getattr(self, name, None)
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
        raise KeyError(f"{name!r} is not a scalar metric")


class EvaluationResult(DomainModel):
    evaluation_id: str
    label: str
    policy_id: str
    checkpoint: str | None = None
    suite: str
    num_tasks: int
    samples_per_task: int
    temperature: float
    overall: MetricsSummary
    by_domain: dict[str, MetricsSummary] = Field(default_factory=dict)
    simulated: bool = False
    created_at: datetime
    code: CodeVersion

    def scope(self, scope: str) -> MetricsSummary | None:
        if scope == "overall":
            return self.overall
        if scope.startswith("domain:"):
            return self.by_domain.get(scope.removeprefix("domain:"))
        raise ValueError(f"unknown metric scope {scope!r}")


class GateRuleKind(StrEnum):
    MAX_REGRESSION = "max_regression"
    """candidate >= baseline - threshold"""
    NO_REGRESSION = "no_regression"
    """candidate >= baseline (threshold acts as a tiny numerical tolerance)"""
    MIN_VALUE = "min_value"
    """candidate >= threshold"""
    MAX_VALUE = "max_value"
    """candidate <= threshold"""


class GateRule(DomainModel):
    name: str
    metric: str
    scope: str = "overall"
    kind: GateRuleKind
    threshold: float = 0.0
    required: bool = True


class GateCheck(DomainModel):
    rule: GateRule
    baseline: float | None
    candidate: float | None
    passed: bool
    detail: str


class GateReport(DomainModel):
    baseline_label: str
    candidate_label: str
    passed: bool
    checks: tuple[GateCheck, ...]
    warnings: tuple[str, ...] = ()


class PromotionState(StrEnum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    REJECTED = "rejected"


class PromotionRecord(DomainModel):
    checkpoint_id: str
    state: PromotionState
    updated_at: datetime
    gate_passed: bool | None = None
    reasons: tuple[str, ...] = ()
    evaluation_id: str | None = None
    simulated: bool = False
