"""Judge calibration: agreement with reference labels and simple bias probes.

Reference labels can come from deterministic verifiers (on tasks where both apply) or from humans.
These statistics describe a judge on a specific label set; they do not certify it in general.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AgreementReport:
    n: int
    accuracy: float
    precision: float | None
    recall: float | None
    f1: float | None
    cohen_kappa: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    judge_pass_rate: float
    reference_pass_rate: float


def binary_agreement(judge: Sequence[bool], reference: Sequence[bool]) -> AgreementReport:
    """Confusion matrix, accuracy, precision/recall/F1 and Cohen's kappa (judge vs reference)."""
    if len(judge) != len(reference):
        raise ValueError("judge and reference label lists must have the same length")
    n = len(judge)
    if n == 0:
        raise ValueError("need at least one label")
    tp = sum(1 for j, r in zip(judge, reference, strict=True) if j and r)
    fp = sum(1 for j, r in zip(judge, reference, strict=True) if j and not r)
    tn = sum(1 for j, r in zip(judge, reference, strict=True) if not j and not r)
    fn = n - tp - fp - tn
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    judge_yes, reference_yes = (tp + fp) / n, (tp + fn) / n
    expected = judge_yes * reference_yes + (1 - judge_yes) * (1 - reference_yes)
    kappa = 1.0 if expected == 1.0 else (accuracy - expected) / (1 - expected)
    return AgreementReport(
        n=n,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        cohen_kappa=kappa,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        judge_pass_rate=judge_yes,
        reference_pass_rate=reference_yes,
    )


def verbosity_flip_rate(
    judge: Callable[[str], bool], responses: Sequence[str], padding: str
) -> float:
    """Fraction of verdicts that change when content-free ``padding`` is appended.

    A robust judge should be insensitive to irrelevant verbosity; a high flip rate indicates a
    length/verbosity bias that an RL policy could exploit.
    """
    if not responses:
        raise ValueError("need at least one response")
    flips = sum(1 for text in responses if judge(text) != judge(f"{text}\n\n{padding}"))
    return flips / len(responses)


def position_flip_rate(
    pairwise_judge: Callable[[str, str], int], pairs: Sequence[tuple[str, str]]
) -> float:
    """Fraction of pairwise preferences that change when the two candidates are swapped.

    ``pairwise_judge(a, b)`` returns 0 if it prefers ``a`` and 1 if it prefers ``b``.
    """
    if not pairs:
        raise ValueError("need at least one pair")
    inconsistent = 0
    for first, second in pairs:
        original = pairwise_judge(first, second)
        swapped = pairwise_judge(second, first)
        if original == swapped:
            inconsistent += 1
    return inconsistent / len(pairs)
