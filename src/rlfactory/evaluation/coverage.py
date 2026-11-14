"""Verifier coverage: how much of a suite is deterministically verifiable vs heuristic vs judged."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.verification import VerifierKind
from rlfactory.errors import RegistryError
from rlfactory.verifiers.registry import VerifierRegistry


@dataclass
class VerifierCoverage:
    total: int
    counts: dict[str, int] = field(default_factory=dict)
    by_domain: dict[str, dict[str, int]] = field(default_factory=dict)

    def fraction(self, kind: str) -> float:
        return self.counts.get(kind, 0) / self.total if self.total else 0.0


def verifier_coverage(
    tasks: Sequence[Task],
    verifiers: VerifierRegistry,
    overrides: Mapping[TaskDomain, str] | None = None,
) -> VerifierCoverage:
    overrides = overrides or {}
    counts: Counter[str] = Counter()
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        name = overrides.get(task.domain, task.verifier)
        try:
            kind = verifiers.get(name).kind.value
        except RegistryError:
            kind = "unverifiable"
        counts[kind] += 1
        by_domain[task.domain.value][kind] += 1
    for kind in (*(k.value for k in VerifierKind), "unverifiable"):
        counts.setdefault(kind, 0)
    return VerifierCoverage(
        total=len(tasks),
        counts=dict(counts),
        by_domain={domain: dict(c) for domain, c in sorted(by_domain.items())},
    )
