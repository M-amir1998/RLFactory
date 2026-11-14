"""Episode bookkeeping shared by the async rollout worker and synchronous task validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from rlfactory.domain.common import CodeVersion, ComponentRef
from rlfactory.domain.model import GenerationConfig
from rlfactory.domain.reward import RewardResult
from rlfactory.domain.task import Task
from rlfactory.domain.trajectory import (
    FailureRecord,
    Lineage,
    Message,
    ModelAction,
    Observation,
    Role,
    StepResult,
    Trajectory,
    TrajectoryMetadata,
    TrajectoryStatus,
)
from rlfactory.domain.verification import VerificationResult
from rlfactory.environments.base import Environment
from rlfactory.tasks.prompts import PromptBuilder

MAX_TURNS = 64


class EpisodeRecorder:
    """Accumulates messages, actions and observations for one episode."""

    def __init__(
        self, task: Task, initial_messages: tuple[Message, ...], first: Observation
    ) -> None:
        self.task = task
        self.messages: list[Message] = list(initial_messages)
        self.actions: list[ModelAction] = []
        self.observations: list[Observation] = [first]
        self.done = False
        self.truncated = False

    @property
    def turn(self) -> int:
        return len(self.actions)

    def record(self, action: ModelAction, result: StepResult) -> None:
        self.actions.append(action)
        self.observations.append(result.observation)
        self.messages.append(Message(role=Role.ASSISTANT, content=action.text))
        self.done = result.done
        self.truncated = self.truncated or result.truncated
        if not result.done:
            self.messages.append(
                Message(role=result.observation.role, content=result.observation.content)
            )

    def build(
        self,
        *,
        trajectory_id: str,
        policy_id: str,
        lineage: Lineage,
        sample_index: int,
        group_id: str,
        started_at: datetime,
        attempts: int = 1,
        status: TrajectoryStatus = TrajectoryStatus.COMPLETED,
        verification: VerificationResult | None = None,
        rewards: RewardResult | None = None,
        failure: FailureRecord | None = None,
    ) -> Trajectory:
        finished = datetime.now(UTC)
        return Trajectory(
            trajectory_id=trajectory_id,
            task_id=self.task.task_id,
            policy_id=policy_id,
            status=status,
            messages=tuple(self.messages),
            actions=tuple(self.actions),
            observations=tuple(self.observations),
            verification=verification,
            rewards=rewards,
            failure=failure,
            lineage=lineage,
            metadata=TrajectoryMetadata(
                domain=self.task.domain,
                task_family=self.task.family,
                sample_index=sample_index,
                group_id=group_id,
                attempts=attempts,
                started_at=started_at,
                finished_at=finished,
                wall_time_s=max(0.0, (finished - started_at).total_seconds()),
                truncated=self.truncated,
            ),
        )


def offline_lineage(
    task: Task,
    *,
    run_id: str,
    environment: ComponentRef,
    verifier: ComponentRef,
    reward: ComponentRef,
    prompt_template: ComponentRef,
    policy_id: str,
    code: CodeVersion,
) -> Lineage:
    """Lineage for episodes that are not produced by a model (validation, oracle replays)."""
    return Lineage(
        run_id=run_id,
        task_version=task.version,
        task_fingerprint=task.fingerprint(),
        environment=environment,
        verifier=verifier,
        reward=reward,
        prompt_template=prompt_template,
        policy_id=policy_id,
        generation=GenerationConfig(temperature=0.0),
        seed=0,
        code=code,
    )


def run_sync_episode(
    environment: Environment,
    task: Task,
    prompts: PromptBuilder,
    respond: Callable[[int, Sequence[Message]], str],
    *,
    max_turns: int = MAX_TURNS,
) -> EpisodeRecorder:
    """Drive an environment with a synchronous responder ``respond(turn, messages) -> text``."""
    first = environment.reset(task)
    recorder = EpisodeRecorder(task, prompts.initial_messages(task, first), first)
    while not recorder.done and recorder.turn < max_turns:
        action = ModelAction(text=respond(recorder.turn, recorder.messages))
        recorder.record(action, environment.step(action))
    return recorder
