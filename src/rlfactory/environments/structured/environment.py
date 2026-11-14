"""Single-turn structured-output environment (JSON transformations, extraction, constraints)."""

from __future__ import annotations

import json

from typing_extensions import override

from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.trajectory import ModelAction, Observation
from rlfactory.environments.base import SingleTurnEnvironment
from rlfactory.environments.parsing import extract_json_answer


class StructuredOutputEnvironment(SingleTurnEnvironment):
    name = "structured_output"
    version = "1.0.0"
    domain = TaskDomain.STRUCTURED_OUTPUT

    @override
    def evaluate(self, task: Task, action: ModelAction) -> Observation:
        raw = extract_json_answer(action.text)
        if raw is None:
            return Observation(
                content="No JSON answer found in <answer> tags.", parse_error="missing_answer_tag"
            )
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            return Observation(
                content=f"Answer is not valid JSON: {exc.msg}.", parse_error="invalid_json"
            )
        return Observation(content="JSON answer received.", parsed_answer=raw)
