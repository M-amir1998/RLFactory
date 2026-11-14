"""Verifier outputs. Verification answers "is this correct?"; rewards are built separately."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from rlfactory.domain.common import ComponentRef, DomainModel


class VerifierKind(StrEnum):
    DETERMINISTIC = "deterministic"
    """Programmatic check with an objective ground truth (exact, numeric, unit tests, schema)."""
    HEURISTIC = "heuristic"
    """Programmatic but approximate check (e.g. a regex standing in for semantics)."""
    MODEL_JUDGE = "model_judge"
    """An LLM judge; never the primary signal in RLFactory."""


class ReasonCode(StrEnum):
    OK = "ok"
    WRONG_ANSWER = "wrong_answer"
    NO_ANSWER = "no_answer"
    INVALID_FORMAT = "invalid_format"
    INVALID_JSON = "invalid_json"
    SCHEMA_VIOLATION = "schema_violation"
    CONSTRAINT_VIOLATION = "constraint_violation"
    VALUE_MISMATCH = "value_mismatch"
    PATTERN_MISMATCH = "pattern_mismatch"
    NO_CODE = "no_code"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    TESTS_FAILED = "tests_failed"
    EXECUTION_REPORT_MISSING = "execution_report_missing"
    TOOL_MISUSE = "tool_misuse"
    REQUIRED_TOOL_NOT_USED = "required_tool_not_used"
    STEP_LIMIT = "step_limit"
    JUDGE_REJECTED = "judge_rejected"
    JUDGE_ERROR = "judge_error"
    COMPONENT_FAILED = "component_failed"
    VERIFIER_ERROR = "verifier_error"


class VerificationComponent(DomainModel):
    """One weighted component of a composite verification."""

    name: str
    role: Literal["correctness", "format", "constraint"]
    weight: float = Field(ge=0.0)
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason_code: str
    required: bool = False


class VerificationResult(DomainModel):
    """Bounded verifier output.

    ``score`` is always in ``[0, 1]``: 1.0 means fully verified correct, 0.0 means verified wrong
    or unverifiable. ``passed`` is the binary decision used for accuracy metrics; it may be stricter
    than ``score > 0`` (e.g. partial unit-test credit does not pass).
    """

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason_code: str
    verifier: ComponentRef
    kind: VerifierKind = VerifierKind.DETERMINISTIC
    components: tuple[VerificationComponent, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _passed_implies_positive_score(self) -> VerificationResult:
        if self.passed and self.score <= 0.0:
            raise ValueError("a passing verification must have a positive score")
        return self
