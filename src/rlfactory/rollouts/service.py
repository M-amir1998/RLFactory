"""Application service: wire registries, policy, store and orchestrator into one call."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rlfactory.domain.task import Task
from rlfactory.environments.base import EnvironmentContext
from rlfactory.environments.registry import EnvironmentRegistry, default_environment_registry
from rlfactory.environments.tools.registry import default_tool_registry
from rlfactory.policies.base import Policy
from rlfactory.rewards.registry import RewardRegistry, RewardSpec
from rlfactory.rollouts.config import RolloutConfig
from rlfactory.rollouts.jobs import plan_jobs
from rlfactory.rollouts.orchestrator import ProgressReporter, RolloutOrchestrator, RolloutSummary
from rlfactory.rollouts.worker import RolloutContext, RolloutWorker
from rlfactory.runtime.git import detect_code_version
from rlfactory.sandbox.base import ResourceLimits, Sandbox
from rlfactory.sandbox.subprocess_sandbox import SubprocessSandbox
from rlfactory.tasks.prompts import PromptBuilder
from rlfactory.telemetry.metrics import RolloutMetrics
from rlfactory.trajectories.store import TrajectoryStore
from rlfactory.verifiers.registry import VerifierRegistry, VerifierSpec


@dataclass(frozen=True)
class RolloutDependencies:
    environments: EnvironmentRegistry
    verifiers: VerifierRegistry
    rewards: RewardRegistry
    prompts: PromptBuilder
    environment_context: EnvironmentContext

    @classmethod
    def default(
        cls,
        *,
        sandbox: Sandbox | None = None,
        python_limits: ResourceLimits | None = None,
        verifier_specs: Mapping[str, VerifierSpec] | None = None,
        reward_specs: Mapping[str, RewardSpec] | None = None,
        prompts: PromptBuilder | None = None,
    ) -> RolloutDependencies:
        return cls(
            environments=default_environment_registry(),
            verifiers=VerifierRegistry(verifier_specs),
            rewards=RewardRegistry(reward_specs),
            prompts=prompts or PromptBuilder(),
            environment_context=EnvironmentContext(
                sandbox=sandbox or SubprocessSandbox(),
                tools=default_tool_registry(),
                python_limits=python_limits or ResourceLimits(),
            ),
        )

    def check(self, tasks: Sequence[Task], config: RolloutConfig) -> None:
        """Fail fast on unknown environments, verifiers or rewards before any generation."""
        self.rewards.get(config.reward)
        for name in {t.environment for t in tasks}:
            self.environments.spec(name)
        for task in tasks:
            self.verifiers.get(config.verifier_overrides.get(task.domain, task.verifier))


async def run_rollouts(
    *,
    run_id: str,
    tasks: Sequence[Task],
    policy: Policy,
    config: RolloutConfig,
    store: TrajectoryStore,
    deps: RolloutDependencies,
    progress: ProgressReporter | None = None,
    cancel: asyncio.Event | None = None,
    metrics: RolloutMetrics | None = None,
) -> RolloutSummary:
    deps.check(tasks, config)
    context = RolloutContext(
        run_id=run_id,
        policy=policy,
        environments=deps.environments,
        environment_context=deps.environment_context,
        verifiers=deps.verifiers,
        rewards=deps.rewards,
        prompts=deps.prompts,
        config=config,
        code=detect_code_version(),
        metrics=metrics or RolloutMetrics(),
    )
    jobs = plan_jobs(tasks, config, policy_id=policy.policy_id, checkpoint=policy.checkpoint)
    orchestrator = RolloutOrchestrator(RolloutWorker(context), store, config, progress=progress)
    return await orchestrator.run(jobs, cancel)
