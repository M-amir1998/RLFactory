"""Single-turn text-transformation environment."""

from __future__ import annotations

from typing_extensions import override

from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.trajectory import ModelAction, Observation
from rlfactory.environments.base import SingleTurnEnvironment
from rlfactory.environments.parsing import ANSWER_TAG, extract_tag


class TransformationEnvironment(SingleTurnEnvironment):
    name = "transformation"
    version = "1.0.0"
    domain = TaskDomain.TRANSFORMATION

    @override
    def evaluate(self, task: Task, action: ModelAction) -> Observation:
        answer = extract_tag(action.text, ANSWER_TAG)
        if answer is None:
            return Observation(content="No <answer> tag found.", parse_error="missing_answer_tag")
        return Observation(content="Answer received.", parsed_answer=answer)
