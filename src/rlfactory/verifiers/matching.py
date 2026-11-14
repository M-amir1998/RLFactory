"""Answer-matching verifiers: exact text, numeric equivalence and regular expressions."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Literal

from typing_extensions import override

from rlfactory.domain.numeric import canonical_number, last_number, parse_number
from rlfactory.domain.task import Task
from rlfactory.domain.trajectory import Trajectory
from rlfactory.domain.verification import ReasonCode, VerificationResult, VerifierKind
from rlfactory.errors import VerifierError
from rlfactory.verifiers.base import BaseVerifier, final_answer, final_parse_error, reference_answer


def normalize_text(text: str, *, case_sensitive: bool, normalize_whitespace: bool) -> str:
    if normalize_whitespace:
        text = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return text if case_sensitive else text.casefold()


class ExactMatchVerifier(BaseVerifier):
    """Compares the extracted answer with the reference text.

    With ``normalize_whitespace=False`` this is the strict verifier used in verifier-quality
    studies: ``"27/2"`` does not match ``"13.5"``.
    """

    def __init__(
        self,
        name: str = "exact_match",
        *,
        case_sensitive: bool = True,
        normalize_whitespace: bool = True,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    @override
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        expected = reference_answer(task)
        if expected is None:
            raise VerifierError(f"{self.name}: task {task.task_id} has no reference answer")
        answer = final_answer(trajectory)
        if answer is None:
            return self.fail(ReasonCode.NO_ANSWER, parse_error=final_parse_error(trajectory))
        normalized = normalize_text(
            answer,
            case_sensitive=self.case_sensitive,
            normalize_whitespace=self.normalize_whitespace,
        )
        target = normalize_text(
            expected,
            case_sensitive=self.case_sensitive,
            normalize_whitespace=self.normalize_whitespace,
        )
        if normalized == target:
            return self.ok()
        return self.fail(ReasonCode.WRONG_ANSWER, expected=expected[:200], got=answer[:200])


def numbers_close(got: Fraction, expected: Fraction, rel_tol: float, abs_tol: float) -> bool:
    if got == expected:
        return True
    tolerance = max(Fraction(rel_tol) * abs(expected), Fraction(abs_tol))
    return abs(got - expected) <= tolerance


class NumericVerifier(BaseVerifier):
    """Exact rational comparison with optional tolerance.

    ``answer_source="parsed_or_last_number"`` falls back to the last number in the raw completion
    when no ``<answer>`` tag was produced. That leniency makes rewards less sparse but also more
    exploitable; it exists for verifier-design studies.
    """

    def __init__(
        self,
        name: str = "numeric",
        *,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-9,
        answer_source: Literal["parsed", "parsed_or_last_number"] = "parsed",
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        if rel_tol < 0 or abs_tol < 0:
            raise ValueError("tolerances must be non-negative")
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.answer_source = answer_source

    @override
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        reference = reference_answer(task)
        expected = parse_number(reference) if reference is not None else None
        if expected is None:
            raise VerifierError(f"{self.name}: task {task.task_id} has no numeric reference")
        answer = final_answer(trajectory)
        got = parse_number(answer) if answer is not None else None
        source = "parsed"
        if got is None and self.answer_source == "parsed_or_last_number":
            got = last_number(trajectory.completion_text)
            source = "last_number"
        if got is None:
            reason = ReasonCode.NO_ANSWER if answer is None else ReasonCode.INVALID_FORMAT
            return self.fail(reason, parse_error=final_parse_error(trajectory))
        if numbers_close(got, expected, self.rel_tol, self.abs_tol):
            return self.ok(answer_source=source)
        return self.fail(
            ReasonCode.WRONG_ANSWER,
            expected=canonical_number(expected),
            got=canonical_number(got),
            answer_source=source,
        )


class RegexVerifier(BaseVerifier):
    """Pattern check on the raw completion or the extracted answer.

    Regexes rarely capture semantics, so this verifier is ``heuristic`` by default; it is marked
    ``deterministic`` only when used as a pure format check.
    """

    def __init__(
        self,
        name: str,
        pattern: str,
        *,
        target: Literal["completion", "answer"] = "completion",
        mode: Literal["search", "fullmatch"] = "search",
        kind: VerifierKind = VerifierKind.HEURISTIC,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        self.pattern = re.compile(pattern, re.DOTALL)
        self.target = target
        self.mode = mode
        self.kind = kind

    @override
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        text = (
            trajectory.completion_text if self.target == "completion" else final_answer(trajectory)
        )
        if text is None:
            return self.fail(ReasonCode.NO_ANSWER)
        matcher = self.pattern.fullmatch if self.mode == "fullmatch" else self.pattern.search
        if matcher(text) is not None:
            return self.ok()
        return self.fail(ReasonCode.PATTERN_MISMATCH)
