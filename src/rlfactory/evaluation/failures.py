"""Failure analysis: observed categories, kept separate from (labelled) speculative hypotheses."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rlfactory.domain.numeric import last_number, parse_number
from rlfactory.domain.result import FailureCategory
from rlfactory.domain.trajectory import FailureKind, Trajectory, TrajectoryStatus
from rlfactory.domain.verification import ReasonCode

_TIMEOUT = {ReasonCode.TIMEOUT.value}
_RUNTIME = {
    ReasonCode.SYNTAX_ERROR.value,
    ReasonCode.RUNTIME_ERROR.value,
    ReasonCode.RESOURCE_LIMIT.value,
    ReasonCode.EXECUTION_REPORT_MISSING.value,
}
_TOOLS = {
    ReasonCode.TOOL_MISUSE.value,
    ReasonCode.REQUIRED_TOOL_NOT_USED.value,
    ReasonCode.STEP_LIMIT.value,
}
_FORMAT = {
    ReasonCode.NO_ANSWER.value,
    ReasonCode.INVALID_FORMAT.value,
    ReasonCode.INVALID_JSON.value,
    ReasonCode.NO_CODE.value,
    ReasonCode.PATTERN_MISMATCH.value,
    ReasonCode.SCHEMA_VIOLATION.value,
}
_VERIFIER = {ReasonCode.JUDGE_ERROR.value, ReasonCode.VERIFIER_ERROR.value}


@dataclass(frozen=True)
class FailureObservation:
    trajectory_id: str
    task_id: str
    domain: str
    family: str
    category: FailureCategory
    reason_code: str | None
    failure_kind: str | None
    error_type: str | None
    reward: float | None
    observed: str
    """What the data shows (factual)."""
    hypothesis: str | None = None
    """Speculative root cause, explicitly labelled; never used for gating."""


def _hypothesis(trajectory: Trajectory, category: FailureCategory) -> str | None:
    observation = trajectory.final_observation
    verification = trajectory.verification
    if (
        category is FailureCategory.FORMAT_ERROR
        and trajectory.metadata.domain.value == "arithmetic"
    ):
        expected = verification.metadata.get("expected") if verification else None
        found = last_number(trajectory.completion_text)
        if isinstance(expected, str) and found is not None and found == parse_number(expected):
            return "SPECULATIVE: correct value present but not in the required <answer> format"
    if category is FailureCategory.REASONING_ERROR and verification is not None:
        expected, got = verification.metadata.get("expected"), verification.metadata.get("got")
        if isinstance(expected, str) and isinstance(got, str):
            e, g = parse_number(expected), parse_number(got)
            if e is not None and g is not None and abs(e - g) <= 2:
                return "SPECULATIVE: near miss (possible arithmetic slip)"
    if category is FailureCategory.TOOL_MISUSE:
        errors = [r.error for r in trajectory.tool_records() if r.error]
        if any("not available" in e or "unknown tool" in e for e in errors):
            return "SPECULATIVE: called a tool that does not exist (hallucinated tool)"
    if category is FailureCategory.CODE_RUNTIME_ERROR and observation and observation.execution:
        return f"SPECULATIVE: program raised {observation.execution.error_type or 'an error'}"
    return None


def classify(trajectory: Trajectory) -> FailureObservation | None:
    """Observed failure category of one trajectory, or ``None`` if it passed."""
    if trajectory.passed:
        return None
    failure = trajectory.failure
    verification = trajectory.verification
    reason = verification.reason_code if verification else None
    execution = trajectory.final_observation.execution if trajectory.final_observation else None
    if trajectory.status is TrajectoryStatus.FAILED:
        kind = failure.kind if failure else FailureKind.INFRASTRUCTURE_ERROR
        category = (
            FailureCategory.VERIFIER_FAILURE
            if kind is FailureKind.VERIFIER_ERROR
            else FailureCategory.INFRASTRUCTURE_FAILURE
        )
        observed = f"episode failed: {kind.value}"
    elif reason in _TIMEOUT or (failure and failure.kind is FailureKind.SANDBOX_TIMEOUT):
        category, observed = FailureCategory.TIMEOUT, "generated program exceeded its time limit"
    elif reason in _RUNTIME:
        category, observed = (
            FailureCategory.CODE_RUNTIME_ERROR,
            f"code did not run cleanly ({reason})",
        )
    elif reason in _TOOLS or any(not r.succeeded for r in trajectory.tool_records()):
        category, observed = FailureCategory.TOOL_MISUSE, f"tool usage problem ({reason})"
    elif reason in _FORMAT:
        category, observed = (
            FailureCategory.FORMAT_ERROR,
            f"output not in the required format ({reason})",
        )
    elif reason in _VERIFIER:
        category, observed = (
            FailureCategory.VERIFIER_FAILURE,
            f"verifier could not decide ({reason})",
        )
    else:
        category, observed = FailureCategory.REASONING_ERROR, f"well-formed but wrong ({reason})"
    return FailureObservation(
        trajectory_id=trajectory.trajectory_id,
        task_id=trajectory.task_id,
        domain=trajectory.metadata.domain.value,
        family=trajectory.metadata.task_family,
        category=category,
        reason_code=reason,
        failure_kind=failure.kind.value if failure else None,
        error_type=execution.error_type if execution else None,
        reward=trajectory.reward_total,
        observed=observed,
        hypothesis=_hypothesis(trajectory, category),
    )


@dataclass
class FailureAnalysis:
    observations: list[FailureObservation]
    counts: dict[str, int] = field(default_factory=dict)
    by_domain: dict[str, dict[str, int]] = field(default_factory=dict)

    def failed_task_ids(self, categories: Sequence[FailureCategory] | None = None) -> list[str]:
        wanted = set(categories) if categories else None
        ordered = dict.fromkeys(
            o.task_id
            for o in self.observations
            if (wanted is None or o.category in wanted)
            and o.category is not FailureCategory.INFRASTRUCTURE_FAILURE
        )
        return list(ordered)


def analyze_failures(trajectories: Iterable[Trajectory]) -> FailureAnalysis:
    observations = [o for t in trajectories if (o := classify(t)) is not None]
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for observation in observations:
        by_domain[observation.domain][observation.category.value] += 1
    return FailureAnalysis(
        observations=observations,
        counts=dict(Counter(o.category.value for o in observations)),
        by_domain={domain: dict(counts) for domain, counts in sorted(by_domain.items())},
    )


def write_failures_parquet(analysis: FailureAnalysis, path: Path) -> None:
    rows = []
    for observation in analysis.observations:
        row = asdict(observation)
        row["category"] = observation.category.value
        rows.append(row)
    schema = pa.schema(
        [
            ("trajectory_id", pa.string()),
            ("task_id", pa.string()),
            ("domain", pa.string()),
            ("family", pa.string()),
            ("category", pa.string()),
            ("reason_code", pa.string()),
            ("failure_kind", pa.string()),
            ("error_type", pa.string()),
            ("reward", pa.float64()),
            ("observed", pa.string()),
            ("hypothesis", pa.string()),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
