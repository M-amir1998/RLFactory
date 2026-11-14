"""Composite verifier: weighted correctness / format / constraint components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from typing_extensions import override

from rlfactory.domain.task import Task
from rlfactory.domain.trajectory import Trajectory
from rlfactory.domain.verification import (
    ReasonCode,
    VerificationComponent,
    VerificationResult,
    VerifierKind,
)
from rlfactory.errors import ConfigError
from rlfactory.verifiers.base import BaseVerifier, Verifier

ComponentRole = Literal["correctness", "format", "constraint"]
_KIND_ORDER = {
    VerifierKind.DETERMINISTIC: 0,
    VerifierKind.HEURISTIC: 1,
    VerifierKind.MODEL_JUDGE: 2,
}


@dataclass(frozen=True)
class CompositeComponent:
    verifier: Verifier
    role: ComponentRole
    weight: float
    required: bool = False


class CompositeVerifier(BaseVerifier):
    """``score = sum(w_i * s_i) / sum(w_i)``; ``passed`` = every required component passed.

    If no component is marked required, all correctness components are required. Correctness
    must carry at least ``min_correctness_share`` of the total weight so that formatting can never
    dominate an RLVR signal by accident (lower it explicitly for ablations).
    """

    def __init__(
        self,
        name: str,
        components: Sequence[CompositeComponent],
        *,
        min_correctness_share: float = 0.5,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        if not components:
            raise ConfigError(f"composite verifier {name!r} needs at least one component")
        if any(component.weight <= 0 for component in components):
            raise ConfigError(f"composite verifier {name!r}: weights must be positive")
        total = sum(component.weight for component in components)
        correctness = sum(c.weight for c in components if c.role == "correctness")
        if correctness / total < min_correctness_share:
            raise ConfigError(
                f"composite verifier {name!r}: correctness carries {correctness / total:.0%} of the "
                f"weight, below the configured minimum {min_correctness_share:.0%}"
            )
        self.components = tuple(components)
        self.total_weight = total
        explicit = [c for c in self.components if c.required]
        self.required = explicit or [c for c in self.components if c.role == "correctness"]
        if not self.required:
            raise ConfigError(f"composite verifier {name!r} has no required component")
        self.kind = max(
            (c.verifier.kind for c in self.components), key=lambda kind: _KIND_ORDER[kind]
        )

    @override
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        results = [
            (component, component.verifier.verify(task, trajectory))
            for component in self.components
        ]
        score = sum(c.weight * r.score for c, r in results) / self.total_weight
        failing = [(c, r) for c, r in results if c in self.required and not r.passed]
        parts = tuple(
            VerificationComponent(
                name=c.verifier.name,
                role=c.role,
                weight=c.weight,
                passed=r.passed,
                score=r.score,
                reason_code=r.reason_code,
                required=c in self.required,
            )
            for c, r in results
        )
        passed = not failing
        reason = ReasonCode.OK.value if passed else failing[0][1].reason_code
        return VerificationResult(
            passed=passed,
            score=min(1.0, max(0.0, score)),
            reason_code=reason,
            verifier=self.ref,
            kind=self.kind,
            components=parts,
        )
