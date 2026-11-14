"""Task validation: every synthetic task must be independently solvable and discriminative.

A task is accepted only if (1) its environment and verifier resolve, (2) the reference solution,
played through the real environment (including the sandbox for Python), passes its verifier, and
(3) at least one known-wrong answer is rejected. Tasks failing any check are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from random import Random

from rlfactory.domain.common import ComponentRef, stable_seed
from rlfactory.domain.task import Task
from rlfactory.domain.trajectory import TrajectoryStatus
from rlfactory.domain.verification import VerificationResult
from rlfactory.environments.base import EnvironmentContext
from rlfactory.environments.registry import EnvironmentRegistry
from rlfactory.errors import RegistryError, RLFactoryError
from rlfactory.policies.oracle import ResponseMode, ResponseOracle
from rlfactory.rollouts.episode import offline_lineage, run_sync_episode
from rlfactory.runtime.git import detect_code_version
from rlfactory.tasks.prompts import PromptBuilder
from rlfactory.verifiers.base import Verifier
from rlfactory.verifiers.registry import VerifierRegistry

_NO_REWARD = ComponentRef(name="none", version="0")


@dataclass(frozen=True)
class ValidationOutcome:
    task_id: str
    valid: bool
    reasons: tuple[str, ...] = ()


class TaskValidator:
    def __init__(
        self,
        environments: EnvironmentRegistry,
        verifiers: VerifierRegistry,
        context: EnvironmentContext,
        *,
        prompts: PromptBuilder | None = None,
        wrong_attempts: int = 3,
    ) -> None:
        self.environments = environments
        self.verifiers = verifiers
        self.context = context
        self.prompts = prompts or PromptBuilder()
        self.wrong_attempts = wrong_attempts
        self._oracle = ResponseOracle()
        self._code = detect_code_version()

    def _play(
        self, task: Task, verifier: Verifier, mode: ResponseMode, variant: int
    ) -> VerificationResult:
        environment = self.environments.create(task.environment, self.context)
        rng = Random(stable_seed(task.task_id, mode.value, variant))
        try:
            recorder = run_sync_episode(
                environment,
                task,
                self.prompts,
                lambda turn, _messages: self._oracle.respond(
                    task, mode, verbose=False, turn=turn, rng=rng, canonical=True
                ),
            )
        finally:
            environment.close()
        spec = self.environments.spec(task.environment)
        trajectory = recorder.build(
            trajectory_id=f"validation-{task.task_id}-{mode.value}-{variant}",
            policy_id="reference-oracle",
            lineage=offline_lineage(
                task,
                run_id="task-validation",
                environment=spec.ref,
                verifier=ComponentRef(name=verifier.name, version=verifier.version),
                reward=_NO_REWARD,
                prompt_template=self.prompts.template,
                policy_id="reference-oracle",
                code=self._code,
            ),
            sample_index=variant,
            group_id=f"validation-{task.task_id}",
            started_at=datetime.now(UTC),
            status=TrajectoryStatus.PENDING,
        )
        return verifier.verify(task, trajectory)

    def validate(self, task: Task) -> ValidationOutcome:
        try:
            spec = self.environments.spec(task.environment)
            verifier = self.verifiers.get(task.verifier)
        except RegistryError as exc:
            return ValidationOutcome(task.task_id, False, (str(exc),))
        if spec.domain is not task.domain:
            return ValidationOutcome(
                task.task_id, False, (f"environment {spec.name} cannot run {task.domain} tasks",)
            )
        reasons: list[str] = []
        try:
            reference = self._play(task, verifier, ResponseMode.CORRECT, 0)
            if not reference.passed:
                reasons.append(f"reference solution failed verification ({reference.reason_code})")
            rejected = any(
                not self._play(task, verifier, ResponseMode.WRONG, variant).passed
                for variant in range(self.wrong_attempts)
            )
            if not rejected:
                reasons.append("verifier accepted every known-wrong answer")
        except RLFactoryError as exc:
            reasons.append(f"validation error: {exc}")
        return ValidationOutcome(task.task_id, not reasons, tuple(reasons))
