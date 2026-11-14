"""Dataset builders: verified trajectories -> SFT, preference (DPO) and prompt-only RL records.

All builders:

* use only trainable trajectories (COMPLETED, verified, rewarded); infrastructure failures never
  become training data;
* drop anything the :class:`ContaminationFilter` flags (evaluation tasks, their mutants, matching
  prompts, held-out families, benchmark IDs);
* are deterministic for a fixed input set and seed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from random import Random
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.datasets.records import (
    BuildResult,
    DatasetMessage,
    PreferenceRecord,
    RecordLineage,
    RLPromptRecord,
    SFTRecord,
    to_dataset_messages,
)
from rlfactory.domain.common import stable_hash, stable_seed
from rlfactory.domain.task import Task
from rlfactory.domain.trajectory import Role, Trajectory
from rlfactory.environments.base import EnvironmentContext
from rlfactory.environments.registry import EnvironmentRegistry
from rlfactory.sandbox.mock import MockSandbox
from rlfactory.tasks.contamination import ContaminationFilter
from rlfactory.tasks.prompts import PromptBuilder
from rlfactory.tasks.sampling import TaskStats


def _lineage(trajectories: Sequence[Trajectory]) -> RecordLineage:
    first = trajectories[0]
    return RecordLineage(
        task_id=first.task_id,
        task_family=first.metadata.task_family,
        domain=first.metadata.domain.value,
        trajectory_ids=tuple(t.trajectory_id for t in trajectories),
        run_ids=tuple(sorted({t.lineage.run_id for t in trajectories})),
        policy_id=first.policy_id,
        checkpoint=first.lineage.checkpoint,
        verifier=first.lineage.verifier.name,
        reward_fn=first.lineage.reward.name,
    )


def _length(trajectory: Trajectory) -> int:
    return trajectory.completion_tokens or sum(len(a.text) for a in trajectory.actions)


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: str, b: str) -> float:
    left, right = _shingles(a), _shingles(b)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


class _Tally(Protocol):
    considered: int

    def exclude(self, reason: str, count: int = 1) -> None: ...


class _Screen:
    """Shared eligibility screening with exclusion bookkeeping."""

    def __init__(
        self, contamination: ContaminationFilter | None, tasks: Mapping[str, Task] | None
    ) -> None:
        self.contamination = contamination or ContaminationFilter()
        self.tasks = tasks or {}

    def contaminated(self, trajectory: Trajectory) -> str | None:
        if self.contamination.empty:
            return None
        task = self.tasks.get(trajectory.task_id)
        if task is not None:
            return self.contamination.reason(task)
        prompt = next((m.content for m in trajectory.messages if m.role is Role.USER), "")
        return self.contamination.reason_for_record(
            trajectory.task_id, prompt, trajectory.metadata.task_family
        )

    def eligible(self, trajectories: Iterable[Trajectory], result: _Tally) -> list[Trajectory]:
        kept: list[Trajectory] = []
        for trajectory in trajectories:
            result.considered += 1
            if not trajectory.is_trainable:
                result.exclude("not_trainable")
                continue
            reason = self.contaminated(trajectory)
            if reason is not None:
                result.exclude(f"contamination:{reason}")
                continue
            kept.append(trajectory)
        return kept


def _group(trajectories: Iterable[Trajectory]) -> dict[str, list[Trajectory]]:
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.metadata.group_id].append(trajectory)
    for members in groups.values():
        members.sort(key=lambda t: t.trajectory_id)
    return dict(sorted(groups.items()))


# --------------------------------------------------------------------------------------- SFT


class SFTSelection(StrEnum):
    ALL_SUCCESSFUL = "all_successful"
    HIGHEST_REWARD = "highest_reward"
    SHORTEST_CORRECT = "shortest_correct"
    DIVERSE_CORRECT = "diverse_correct"


class SFTBuilderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: SFTSelection = SFTSelection.SHORTEST_CORRECT
    max_per_task: int = Field(default=1, ge=1)
    min_reward: float | None = None
    diversity_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    """Diverse selection skips completions whose 3-gram Jaccard similarity exceeds this."""
    expand_multi_turn: bool = True
    """Split multi-turn trajectories into one prompt->completion record per assistant turn."""


class SFTDatasetBuilder:
    def __init__(
        self,
        config: SFTBuilderConfig | None = None,
        *,
        contamination: ContaminationFilter | None = None,
        tasks: Mapping[str, Task] | None = None,
    ) -> None:
        self.config = config or SFTBuilderConfig()
        self.screen = _Screen(contamination, tasks)

    def _select(self, passed: list[Trajectory]) -> list[Trajectory]:
        c = self.config
        if c.selection is SFTSelection.HIGHEST_REWARD:
            return sorted(
                passed, key=lambda t: (-(t.reward_total or 0.0), _length(t), t.trajectory_id)
            )[:1]
        if c.selection is SFTSelection.SHORTEST_CORRECT:
            return sorted(passed, key=lambda t: (_length(t), t.trajectory_id))[:1]
        ordered = sorted(passed, key=lambda t: (-(t.reward_total or 0.0), t.trajectory_id))
        chosen: list[Trajectory] = []
        for candidate in ordered:
            text = candidate.completion_text
            if c.selection is SFTSelection.DIVERSE_CORRECT:
                if any(
                    jaccard(text, other.completion_text) >= c.diversity_threshold
                    for other in chosen
                ):
                    continue
            elif any(text == other.completion_text for other in chosen):
                continue
            chosen.append(candidate)
            if len(chosen) >= c.max_per_task:
                break
        return chosen

    def _records(self, trajectory: Trajectory) -> list[SFTRecord]:
        lineage = _lineage([trajectory])
        messages = list(trajectory.messages)
        assistant_positions = [i for i, m in enumerate(messages) if m.role is Role.ASSISTANT]
        if not self.config.expand_multi_turn:
            assistant_positions = assistant_positions[-1:]
        records = []
        for turn, position in enumerate(assistant_positions):
            prompt = to_dataset_messages(messages[:position])
            completion = (DatasetMessage(role="assistant", content=messages[position].content),)
            records.append(
                SFTRecord(
                    record_id=f"sft-{stable_hash(trajectory.trajectory_id, turn, length=16)}",
                    prompt=prompt,
                    completion=completion,
                    reward=trajectory.reward_total,
                    score=trajectory.verification.score if trajectory.verification else None,
                    turn=turn,
                    lineage=lineage,
                )
            )
        return records

    def build(self, trajectories: Iterable[Trajectory]) -> BuildResult[SFTRecord]:
        result: BuildResult[SFTRecord] = BuildResult(records=[])
        eligible = self.screen.eligible(trajectories, result)
        by_task: dict[str, list[Trajectory]] = defaultdict(list)
        for trajectory in eligible:
            if not trajectory.passed:
                result.exclude("not_passed")
                continue
            if (
                self.config.min_reward is not None
                and (trajectory.reward_total or 0.0) < self.config.min_reward
            ):
                result.exclude("below_min_reward")
                continue
            by_task[trajectory.task_id].append(trajectory)
        for task_id in sorted(by_task):
            selected = self._select(by_task[task_id])
            result.exclude("not_selected", len(by_task[task_id]) - len(selected))
            for trajectory in selected:
                result.records.extend(self._records(trajectory))
        return result


# -------------------------------------------------------------------------------- preference


class PreferenceStrategy(StrEnum):
    CORRECT_VS_INCORRECT = "correct_vs_incorrect"
    REWARD_MARGIN = "reward_margin"
    SHORT_CORRECT_VS_VERBOSE_INCORRECT = "short_correct_vs_verbose_incorrect"


class PreferenceBuilderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: PreferenceStrategy = PreferenceStrategy.CORRECT_VS_INCORRECT
    max_pairs_per_task: int = Field(default=2, ge=1)
    min_margin: float = Field(default=0.5, gt=0.0)
    seed: int = 0
    single_turn_only: bool = True
    """Multi-turn pairs would train on tool-result tokens; excluded until turn-level pairing."""


class PreferenceDatasetBuilder:
    """Pairs come only from the same rollout group (same task, same rendered prompt)."""

    def __init__(
        self,
        config: PreferenceBuilderConfig | None = None,
        *,
        contamination: ContaminationFilter | None = None,
        tasks: Mapping[str, Task] | None = None,
    ) -> None:
        self.config = config or PreferenceBuilderConfig()
        self.screen = _Screen(contamination, tasks)

    def _pairs(self, members: list[Trajectory], rng: Random) -> list[tuple[Trajectory, Trajectory]]:
        c = self.config
        passed = [t for t in members if t.passed]
        failed = [t for t in members if not t.passed]
        if c.strategy is PreferenceStrategy.CORRECT_VS_INCORRECT:
            rng.shuffle(passed)
            rng.shuffle(failed)
            candidates = [(a, b) for a in passed for b in failed]
        elif c.strategy is PreferenceStrategy.REWARD_MARGIN:
            ordered = sorted(members, key=lambda t: (-(t.reward_total or 0.0), t.trajectory_id))
            candidates = [
                (hi, lo)
                for i, hi in enumerate(ordered)
                for lo in ordered[i + 1 :]
                if (hi.reward_total or 0.0) - (lo.reward_total or 0.0) >= c.min_margin
            ]
        else:
            short = sorted(passed, key=lambda t: (_length(t), t.trajectory_id))
            verbose = sorted(failed, key=lambda t: (-_length(t), t.trajectory_id))
            candidates = [(a, b) for a in short for b in verbose if _length(b) > _length(a)]
        pairs: list[tuple[Trajectory, Trajectory]] = []
        used: set[tuple[str, str]] = set()
        for chosen, rejected in candidates:
            key = (chosen.completion_text, rejected.completion_text)
            if chosen.completion_text == rejected.completion_text or key in used:
                continue
            used.add(key)
            pairs.append((chosen, rejected))
            if len(pairs) >= c.max_pairs_per_task:
                break
        return pairs

    def build(self, trajectories: Iterable[Trajectory]) -> BuildResult[PreferenceRecord]:
        result: BuildResult[PreferenceRecord] = BuildResult(records=[])
        eligible = self.screen.eligible(trajectories, result)
        for group_id, members in _group(eligible).items():
            if self.config.single_turn_only and any(len(t.actions) > 1 for t in members):
                result.exclude("multi_turn", len(members))
                continue
            prompts = {tuple(t.prompt_messages()) for t in members}
            if len(prompts) != 1:
                result.exclude("prompt_mismatch", len(members))
                continue
            rng = Random(stable_seed(self.config.seed, group_id))
            pairs = self._pairs(members, rng)
            if not pairs:
                result.exclude("no_valid_pair_in_group")
                continue
            prompt = to_dataset_messages(members[0].prompt_messages())
            for chosen, rejected in pairs:
                result.records.append(
                    PreferenceRecord(
                        record_id=f"pref-{stable_hash(chosen.trajectory_id, rejected.trajectory_id, length=16)}",
                        prompt=prompt,
                        chosen=(DatasetMessage(role="assistant", content=chosen.completion_text),),
                        rejected=(
                            DatasetMessage(role="assistant", content=rejected.completion_text),
                        ),
                        chosen_reward=chosen.reward_total,
                        rejected_reward=rejected.reward_total,
                        chosen_passed=chosen.passed,
                        rejected_passed=rejected.passed,
                        strategy=self.config.strategy.value,
                        lineage=_lineage([chosen, rejected]),
                    )
                )
        return result


# ------------------------------------------------------------------------------- GRPO / RLVR


class RLBuilderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reward: str = "binary"
    drop_saturated: bool = True
    """Drop tasks the current policy always or never solves (zero GRPO advantage variance)."""
    min_attempts_for_filter: int = Field(default=2, ge=1)
    single_turn_only: bool = True
    """Multi-step (tool-use) tasks need full rollouts, not single-completion reward functions."""
    max_prompt_chars: int | None = Field(default=None, ge=1)
    """TRL >= 0.26 skips over-long prompts instead of truncating; filter them here instead."""


class RLPromptDatasetBuilder:
    """Prompt-only records plus the full task (for verifier-backed reward functions)."""

    def __init__(
        self,
        config: RLBuilderConfig | None = None,
        *,
        environments: EnvironmentRegistry,
        prompts: PromptBuilder | None = None,
        contamination: ContaminationFilter | None = None,
    ) -> None:
        self.config = config or RLBuilderConfig()
        self.environments = environments
        self.prompts = prompts or PromptBuilder()
        self.contamination = contamination or ContaminationFilter()
        self._context = EnvironmentContext(sandbox=MockSandbox())

    def render_prompt(self, task: Task) -> tuple[DatasetMessage, ...]:
        environment = self.environments.create(task.environment, self._context)
        try:
            first = environment.reset(task)
        finally:
            environment.close()
        return to_dataset_messages(self.prompts.initial_messages(task, first))

    def build(
        self, tasks: Sequence[Task], stats: Mapping[str, TaskStats] | None = None
    ) -> BuildResult[RLPromptRecord]:
        result: BuildResult[RLPromptRecord] = BuildResult(records=[])
        stats = stats or {}
        for task in sorted(tasks, key=lambda t: t.task_id):
            result.considered += 1
            reason = self.contamination.reason(task)
            if reason is not None:
                result.exclude(f"contamination:{reason}")
                continue
            if self.config.single_turn_only and self.environments.spec(task.environment).multi_step:
                result.exclude("multi_step_task")
                continue
            task_stats = stats.get(task.task_id)
            pass_rate = None
            if task_stats is not None and task_stats.attempts:
                pass_rate = task_stats.passes / task_stats.attempts
                saturated = pass_rate in (0.0, 1.0)
                if (
                    self.config.drop_saturated
                    and saturated
                    and task_stats.attempts >= self.config.min_attempts_for_filter
                ):
                    result.exclude(
                        "saturated_all_pass" if pass_rate == 1.0 else "saturated_all_fail"
                    )
                    continue
            prompt = self.render_prompt(task)
            if (
                self.config.max_prompt_chars is not None
                and sum(len(m.content) for m in prompt) > self.config.max_prompt_chars
            ):
                result.exclude("prompt_too_long")
                continue
            result.records.append(
                RLPromptRecord(
                    record_id=f"rl-{stable_hash(task.task_id, self.config.reward, length=16)}",
                    prompt=prompt,
                    task=task,
                    reward=self.config.reward,
                    verifier=task.verifier,
                    pass_rate=pass_rate,
                    lineage=RecordLineage(
                        task_id=task.task_id, task_family=task.family, domain=task.domain.value
                    ),
                )
            )
        return result
