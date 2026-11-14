"""Framework-neutral completion scoring: the bridge every RL framework's reward hook calls.

TRL reward functions, verl ``compute_score`` and OpenRLHF ``reward_func`` all reduce to
"score this completion for this task". Scoring replays the completion through the task's real
environment (sandbox included) and the configured verifier and reward function, exactly as rollouts
do, so training rewards and evaluation verdicts can never silently diverge.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import UTC, datetime

from rlfactory.domain.common import ComponentRef, stable_hash
from rlfactory.domain.reward import RewardResult
from rlfactory.domain.task import Task
from rlfactory.domain.trajectory import ModelAction, TrajectoryStatus
from rlfactory.domain.verification import VerificationResult
from rlfactory.errors import RLFactoryError
from rlfactory.rollouts.episode import EpisodeRecorder, offline_lineage
from rlfactory.rollouts.service import RolloutDependencies
from rlfactory.runtime.git import detect_code_version


class UnsupportedTaskError(RLFactoryError):
    """Multi-step tasks cannot be scored from a single completion."""


class CompletionScorer:
    def __init__(self, deps: RolloutDependencies | None = None, *, cache_size: int = 4096) -> None:
        self.deps = deps or RolloutDependencies.default()
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[VerificationResult, RewardResult]] = OrderedDict()
        self._lock = threading.Lock()
        self._code = detect_code_version()

    def score(
        self, task: Task, completion: str, reward: str = "binary"
    ) -> tuple[VerificationResult, RewardResult]:
        spec = self.deps.environments.spec(task.environment)
        if spec.multi_step:
            raise UnsupportedTaskError(f"{task.task_id}: multi-step tasks need full rollouts")
        key = stable_hash(
            task.task_id, task.fingerprint(), task.verifier, reward, completion, length=24
        )
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        verifier = self.deps.verifiers.get(task.verifier)
        reward_fn = self.deps.rewards.get(reward)
        environment = self.deps.environments.create(task.environment, self.deps.environment_context)
        try:
            first = environment.reset(task)
            recorder = EpisodeRecorder(task, self.deps.prompts.initial_messages(task, first), first)
            action = ModelAction(text=completion)
            recorder.record(action, environment.step(action))
        finally:
            environment.close()
        pending = recorder.build(
            trajectory_id=f"score-{key}",
            policy_id="external-trainer",
            lineage=offline_lineage(
                task,
                run_id="training-reward",
                environment=spec.ref,
                verifier=ComponentRef(name=verifier.name, version=verifier.version),
                reward=ComponentRef(name=reward_fn.name, version=reward_fn.version),
                prompt_template=self.deps.prompts.template,
                policy_id="external-trainer",
                code=self._code,
            ),
            sample_index=0,
            group_id="training-reward",
            started_at=datetime.now(UTC),
            status=TrajectoryStatus.PENDING,
        )
        verification = verifier.verify(task, pending)
        result = (verification, reward_fn.compute(task, pending, verification))
        with self._lock:
            self._cache[key] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result
