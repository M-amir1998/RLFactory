"""Structured-output, unit-test and tool-outcome verifiers."""

from __future__ import annotations

import json
from fractions import Fraction

import jsonschema
from pydantic import JsonValue
from typing_extensions import override

from rlfactory.domain.numeric import ExpressionError, evaluate_condition, parse_number
from rlfactory.domain.task import StructuredOutputPayload, Task, ToolUsePayload
from rlfactory.domain.trajectory import ExecutionStatus, Trajectory
from rlfactory.domain.verification import ReasonCode, VerificationResult
from rlfactory.errors import VerifierError
from rlfactory.verifiers.base import BaseVerifier, final_answer, final_parse_error, json_equal
from rlfactory.verifiers.matching import numbers_close


class JSONSchemaVerifier(BaseVerifier):
    """Valid JSON -> schema-conformant -> expected value and/or constraints satisfied."""

    def __init__(
        self,
        name: str = "json_schema",
        *,
        check_expected: bool = True,
        check_constraints: bool = True,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        self.check_expected = check_expected
        self.check_constraints = check_constraints

    @override
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        payload = task.payload
        if not isinstance(payload, StructuredOutputPayload):
            raise VerifierError(f"{self.name} requires a structured-output task")
        raw = final_answer(trajectory)
        if raw is None:
            reason = (
                ReasonCode.INVALID_JSON
                if final_parse_error(trajectory) == "invalid_json"
                else ReasonCode.NO_ANSWER
            )
            return self.fail(reason, parse_error=final_parse_error(trajectory))
        try:
            answer = json.loads(raw)
        except json.JSONDecodeError:
            return self.fail(ReasonCode.INVALID_JSON)
        try:
            validator = jsonschema.Draft202012Validator(payload.json_schema)
            validator.check_schema(payload.json_schema)
        except jsonschema.SchemaError as exc:
            raise VerifierError(f"invalid schema on task {task.task_id}: {exc.message}") from exc
        errors = sorted(validator.iter_errors(answer), key=lambda error: list(error.path))
        if errors:
            return self.fail(ReasonCode.SCHEMA_VIOLATION, schema_error=errors[0].message[:200])
        if (
            self.check_expected
            and payload.expected is not None
            and not json_equal(answer, payload.expected)
        ):
            return self.fail(ReasonCode.VALUE_MISMATCH)
        if self.check_constraints and payload.constraints:
            violated = self._violated_constraints(answer, payload.constraints)
            if violated:
                return self.fail(ReasonCode.CONSTRAINT_VIOLATION, violated=violated)
        return self.ok()

    @staticmethod
    def _violated_constraints(answer: object, constraints: tuple[str, ...]) -> list[JsonValue]:
        if not isinstance(answer, dict):
            return list[JsonValue](constraints)
        variables = {
            key: Fraction(value)
            for key, value in answer.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        violated: list[JsonValue] = []
        for constraint in constraints:
            try:
                if not evaluate_condition(constraint, variables):
                    violated.append(constraint)
            except ExpressionError:
                violated.append(constraint)
        return violated


_RESOURCE_STATUSES = {
    ExecutionStatus.MEMORY_EXCEEDED,
    ExecutionStatus.OUTPUT_LIMIT,
    ExecutionStatus.FILE_LIMIT,
}


class UnitTestVerifier(BaseVerifier):
    """Scores the hidden-test report recorded by the Python environment.

    ``score`` is the passed fraction when ``partial_credit`` is on; ``passed`` always requires every
    hidden test to pass.
    """

    def __init__(
        self,
        name: str = "unit_test",
        *,
        partial_credit: bool = True,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        self.partial_credit = partial_credit

    @override
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        observation = trajectory.final_observation
        if observation is None or observation.parse_error is not None:
            return self.fail(ReasonCode.NO_CODE, parse_error=final_parse_error(trajectory))
        report = observation.execution
        if report is None:
            return self.fail(ReasonCode.EXECUTION_REPORT_MISSING)
        fraction = report.tests_passed / report.tests_total if report.tests_total else 0.0
        partial = fraction if self.partial_credit else 0.0
        details: dict[str, int | str | None] = {
            "tests_passed": report.tests_passed,
            "tests_total": report.tests_total,
            "execution_status": report.status.value,
            "error_type": report.error_type,
        }
        if report.error_type in {"SyntaxError", "IndentationError", "TabError"}:
            return self.fail(ReasonCode.SYNTAX_ERROR, **details)
        if report.status is ExecutionStatus.TIMEOUT:
            return self.fail(ReasonCode.TIMEOUT, partial, **details)
        if report.status in _RESOURCE_STATUSES:
            return self.fail(ReasonCode.RESOURCE_LIMIT, partial, **details)
        if not report.report_found or report.status is not ExecutionStatus.OK:
            return self.fail(ReasonCode.RUNTIME_ERROR, partial, **details)
        if report.tests_total > 0 and report.tests_passed == report.tests_total:
            return self.ok(**details)
        return self.fail(ReasonCode.TESTS_FAILED, partial, **details)


class ToolOutcomeVerifier(BaseVerifier):
    """Checks the final answer of a tool-use episode and (optionally) required tool usage.

    Requiring tools guards against answer guessing: task facts live only behind the tools.
    """

    def __init__(
        self,
        name: str = "tool_outcome",
        *,
        require_tools: bool = True,
        rel_tol: float = 1e-9,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        self.require_tools = require_tools
        self.rel_tol = rel_tol

    @override
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        payload = task.payload
        if not isinstance(payload, ToolUsePayload):
            raise VerifierError(f"{self.name} requires a tool-use task")
        records = trajectory.tool_records()
        used = {record.call.name for record in records if record.call and record.succeeded}
        errors = sum(1 for record in records if not record.succeeded)
        stats: dict[str, int] = {"tool_calls": len(records), "tool_errors": errors}
        answer = final_answer(trajectory)
        if answer is None:
            reason = (
                ReasonCode.STEP_LIMIT if trajectory.metadata.truncated else ReasonCode.NO_ANSWER
            )
            return self.fail(reason, **stats)
        expected_number = parse_number(payload.expected_answer)
        if expected_number is not None:
            got = parse_number(answer)
            correct = got is not None and numbers_close(
                got, expected_number, self.rel_tol, self.rel_tol
            )
        else:
            correct = answer.strip() == payload.expected_answer.strip()
        if not correct:
            return self.fail(ReasonCode.WRONG_ANSWER, **stats)
        missing = sorted(set(payload.required_tools) - used) if self.require_tools else []
        if missing:
            return self.fail(ReasonCode.REQUIRED_TOOL_NOT_USED, missing=",".join(missing), **stats)
        return self.ok(**stats)
