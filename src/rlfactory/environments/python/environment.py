"""Single-turn Python coding environment backed by a sandbox."""

from __future__ import annotations

from typing_extensions import override

from rlfactory.domain.task import PythonFunctionPayload, Task, TaskDomain
from rlfactory.domain.trajectory import ModelAction, Observation
from rlfactory.environments.base import SingleTurnEnvironment
from rlfactory.environments.parsing import extract_code_block
from rlfactory.environments.python.harness import run_unit_tests


class PythonEnvironment(SingleTurnEnvironment):
    name = "python"
    version = "1.0.0"
    domain = TaskDomain.PYTHON

    @override
    def evaluate(self, task: Task, action: ModelAction) -> Observation:
        payload = task.payload
        if not isinstance(payload, PythonFunctionPayload):  # pragma: no cover - guarded by reset
            raise TypeError("python environment requires a PythonFunctionPayload")
        code = extract_code_block(action.text, "python")
        if code is None:
            return Observation(
                content="No ```python code block found.", parse_error="missing_code_block"
            )
        report = run_unit_tests(self.context.sandbox, code, payload, self.context.python_limits)
        # Only the status is shown; hidden test details stay in the typed side-channel.
        return Observation(
            content=f"Execution finished with status {report.status.value}.",
            parsed_answer=code,
            execution=report,
        )
