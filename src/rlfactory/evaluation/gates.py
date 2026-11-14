"""Config-driven regression gates comparing a candidate evaluation with a baseline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.domain.result import (
    EvaluationResult,
    GateCheck,
    GateReport,
    GateRule,
    GateRuleKind,
)

_EPSILON = 1e-12


class GateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: tuple[GateRule, ...] = Field(default_factory=tuple)
    min_samples_warning: int = Field(default=100, ge=1)


DEFAULT_GATES = GateConfig(
    rules=(
        GateRule(
            name="overall accuracy may not drop more than 2 points",
            metric="verified_accuracy",
            kind=GateRuleKind.MAX_REGRESSION,
            threshold=0.02,
        ),
        GateRule(
            name="python pass rate may not regress",
            metric="verified_accuracy",
            scope="domain:python",
            kind=GateRuleKind.NO_REGRESSION,
        ),
        GateRule(
            name="format compliance stays above 90%",
            metric="format_compliance",
            kind=GateRuleKind.MIN_VALUE,
            threshold=0.9,
        ),
        GateRule(
            name="infrastructure failure rate below 5%",
            metric="infra_failure_rate",
            kind=GateRuleKind.MAX_VALUE,
            threshold=0.05,
        ),
    )
)


def _value(result: EvaluationResult, rule: GateRule) -> float | None:
    summary = result.scope(rule.scope)
    return summary.metric(rule.metric) if summary is not None else None


def _check(rule: GateRule, baseline: float | None, candidate: float | None) -> GateCheck:
    if candidate is None:
        return GateCheck(
            rule=rule,
            baseline=baseline,
            candidate=None,
            passed=False,
            detail="metric unavailable for candidate",
        )
    if rule.kind is GateRuleKind.MIN_VALUE:
        ok = candidate >= rule.threshold - _EPSILON
        detail = f"{candidate:.4f} >= {rule.threshold:.4f}"
    elif rule.kind is GateRuleKind.MAX_VALUE:
        ok = candidate <= rule.threshold + _EPSILON
        detail = f"{candidate:.4f} <= {rule.threshold:.4f}"
    else:
        if baseline is None:
            return GateCheck(
                rule=rule,
                baseline=None,
                candidate=candidate,
                passed=False,
                detail="baseline metric unavailable",
            )
        allowed = (
            rule.threshold if rule.kind is GateRuleKind.MAX_REGRESSION else max(rule.threshold, 0.0)
        )
        ok = candidate >= baseline - allowed - _EPSILON
        detail = f"{candidate:.4f} vs baseline {baseline:.4f} (allowed drop {allowed:.4f})"
    return GateCheck(rule=rule, baseline=baseline, candidate=candidate, passed=ok, detail=detail)


def evaluate_gates(
    baseline: EvaluationResult, candidate: EvaluationResult, config: GateConfig
) -> GateReport:
    checks = tuple(
        _check(rule, _value(baseline, rule), _value(candidate, rule)) for rule in config.rules
    )
    warnings: list[str] = []
    completed = candidate.overall.num_completed
    if completed < config.min_samples_warning:
        ci = candidate.overall.accuracy_ci95
        width = f" (95% CI {ci[0]:.2f}-{ci[1]:.2f})" if ci else ""
        warnings.append(
            f"only {completed} completed candidate trajectories{width}; small differences are noise"
        )
    if baseline.suite != candidate.suite:
        warnings.append(
            f"baseline suite {baseline.suite!r} differs from candidate suite {candidate.suite!r}"
        )
    if candidate.simulated or baseline.simulated:
        warnings.append(
            "simulated evaluation (mock policy/backend): not evidence about real models"
        )
    passed = all(check.passed for check in checks if check.rule.required)
    return GateReport(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        passed=passed,
        checks=checks,
        warnings=tuple(warnings),
    )
