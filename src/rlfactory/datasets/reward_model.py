"""Optional reward-model support: dataset building from preferences and pairwise evaluation.

v0.1 does not train a reward model. The evaluator measures any scorer (a trained RM, a judge, or a
deterministic verifier used as a sanity baseline) on held-out preference pairs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rlfactory.datasets.adapters import to_trl_reward
from rlfactory.datasets.records import DatasetMessage, PreferenceRecord

Scorer = Callable[[tuple[DatasetMessage, ...], tuple[DatasetMessage, ...]], float]
"""``scorer(prompt, completion) -> scalar``; higher means better."""


class RewardModelDatasetBuilder:
    """Reuses verified preference pairs; only pairs with a verified winner are kept by default."""

    def __init__(self, *, require_verified_winner: bool = True) -> None:
        self.require_verified_winner = require_verified_winner

    def build(self, records: Sequence[PreferenceRecord]) -> list[PreferenceRecord]:
        if not self.require_verified_winner:
            return list(records)
        return [r for r in records if r.chosen_passed and not r.rejected_passed]

    @staticmethod
    def to_rows(records: Sequence[PreferenceRecord]) -> list[dict[str, object]]:
        return [to_trl_reward(record) for record in records]


@dataclass(frozen=True)
class RewardModelEvaluation:
    pairs: int
    accuracy: float
    """Fraction of pairs where score(chosen) > score(rejected) (ties count as 0.5)."""
    mean_margin: float


class RewardModelEvaluator:
    def evaluate(
        self, scorer: Scorer, records: Sequence[PreferenceRecord]
    ) -> RewardModelEvaluation:
        if not records:
            raise ValueError("need at least one preference pair")
        wins = 0.0
        margins: list[float] = []
        for record in records:
            chosen = scorer(record.prompt, record.chosen)
            rejected = scorer(record.prompt, record.rejected)
            margins.append(chosen - rejected)
            wins += 1.0 if chosen > rejected else 0.5 if chosen == rejected else 0.0
        return RewardModelEvaluation(
            pairs=len(records),
            accuracy=wins / len(records),
            mean_margin=sum(margins) / len(margins),
        )
