"""Verifier protocol, shared base class and answer-access helpers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Protocol

from pydantic import JsonValue

from rlfactory.domain.common import ComponentRef, canonical_json
from rlfactory.domain.task import (
    ArithmeticPayload,
    StructuredOutputPayload,
    Task,
    ToolUsePayload,
    TransformationPayload,
)
from rlfactory.domain.trajectory import Trajectory
from rlfactory.domain.verification import ReasonCode, VerificationResult, VerifierKind


class Verifier(Protocol):
    name: str
    version: str
    kind: VerifierKind

    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        """Return a bounded result. Must raise :class:`rlfactory.errors.VerifierError` (never
        silently score 0) when the verifier itself cannot operate."""
        ...


class BaseVerifier(ABC):
    implementation_version: str = "1.0.0"
    kind: VerifierKind = VerifierKind.DETERMINISTIC

    def __init__(self, name: str, config_fingerprint: str = "default") -> None:
        self.name = name
        # The configuration hash is part of the version so lineage changes when settings change.
        self.version = f"{self.implementation_version}+{config_fingerprint}"

    @property
    def ref(self) -> ComponentRef:
        return ComponentRef(name=self.name, version=self.version)

    # Positional-only parameters keep ``**metadata`` free of name collisions.
    def result(
        self, passed: bool, score: float, reason: ReasonCode | str, /, **metadata: JsonValue
    ) -> VerificationResult:
        return VerificationResult(
            passed=passed,
            score=min(1.0, max(0.0, score)),
            reason_code=str(reason),
            verifier=self.ref,
            kind=self.kind,
            metadata=metadata,
        )

    def ok(self, score: float = 1.0, /, **metadata: JsonValue) -> VerificationResult:
        return self.result(True, score, ReasonCode.OK, **metadata)

    def fail(
        self, reason: ReasonCode | str, score: float = 0.0, /, **metadata: JsonValue
    ) -> VerificationResult:
        return self.result(False, score, reason, **metadata)

    @abstractmethod
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult: ...


def reference_answer(task: Task) -> str | None:
    """Ground-truth answer text for tasks that have a single correct answer."""
    payload = task.payload
    if isinstance(payload, ArithmeticPayload):
        return payload.answer
    if isinstance(payload, TransformationPayload):
        return payload.expected_output
    if isinstance(payload, ToolUsePayload):
        return payload.expected_answer
    if isinstance(payload, StructuredOutputPayload):
        return canonical_json(payload.expected) if payload.expected is not None else None
    # Python tasks are verified by hidden tests, not a single reference answer.
    return None


def final_answer(trajectory: Trajectory) -> str | None:
    observation = trajectory.final_observation
    return observation.parsed_answer if observation is not None else None


def final_parse_error(trajectory: Trajectory) -> str | None:
    observation = trajectory.final_observation
    return observation.parse_error if observation is not None else "no_action"


def json_equal(left: JsonValue, right: JsonValue, rel_tol: float = 1e-9) -> bool:
    """Structural JSON equality: numbers compare by value, booleans never equal numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(left, right, rel_tol=rel_tol, abs_tol=rel_tol)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_equal(a, b, rel_tol) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            json_equal(left[key], right[key], rel_tol) for key in left
        )
    return type(left) is type(right) and left == right
