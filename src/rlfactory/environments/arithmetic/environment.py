"""Single-turn arithmetic environment.

The environment only extracts the answer; deciding numeric equivalence is the verifier's job, so the
same trajectories can be re-scored under strict or lenient verifiers.
"""

from __future__ import annotations

from typing_extensions import override

from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.trajectory import ModelAction, Observation
from rlfactory.environments.base import SingleTurnEnvironment
from rlfactory.environments.parsing import ANSWER_TAG, extract_tag


class ArithmeticEnvironment(SingleTurnEnvironment):
    name = "arithmetic"
    version = "1.0.0"
    domain = TaskDomain.ARITHMETIC

    @override
    def evaluate(self, task: Task, action: ModelAction) -> Observation:
        answer = extract_tag(action.text, ANSWER_TAG)
        if answer is None:
            return Observation(content="No <answer> tag found.", parse_error="missing_answer_tag")
        if not answer:
            return Observation(content="The <answer> tag is empty.", parse_error="empty_answer")
        return Observation(content="Answer received.", parsed_answer=answer)
