"""Hypothesis property tests for invariants that must hold for all inputs."""

from __future__ import annotations

import itertools
from fractions import Fraction
from random import Random

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rlfactory.domain.common import stable_hash
from rlfactory.domain.numeric import canonical_number, parse_number
from rlfactory.domain.trajectory import Trajectory
from rlfactory.evaluation.metrics import pass_at_k, wilson_interval
from rlfactory.rewards.functions import (
    BinaryReward,
    CompositeReward,
    FormatReward,
    LengthPenalty,
    ToolCostPenalty,
    WeightedVerifierReward,
)
from rlfactory.tasks.generators.catalog import default_template_registry
from rlfactory.tasks.mutation import TaskMutator
from rlfactory.tasks.splits import Split, SplitConfig, assign_split, split_tasks
from rlfactory.verifiers.matching import numbers_close
from tests.helpers import make_task, play

TEMPLATES = default_template_registry()
FAMILIES = [f for f in TEMPLATES.families() if not f.startswith("python.")]
PROPERTY = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])

fractions = st.fractions(min_value=-(10**6), max_value=10**6, max_denominator=10**4)


@PROPERTY
@given(fractions)
def test_canonical_number_roundtrips(value: Fraction) -> None:
    assert parse_number(canonical_number(value)) == value


@PROPERTY
@given(fractions, st.floats(min_value=0, max_value=0.1))
def test_numbers_close_is_reflexive_and_symmetric_at_zero_tolerance(
    value: Fraction, tolerance: float
) -> None:
    assert numbers_close(value, value, tolerance, tolerance)
    assert numbers_close(value, value + 1, 0.0, 0.0) is False


@PROPERTY
@given(st.sampled_from(FAMILIES), st.integers(0, 10**6), st.integers(1, 5))
def test_generated_tasks_are_deterministic_and_serialisable(
    family: str, seed: int, difficulty: int
) -> None:
    template = TEMPLATES.get(family)
    task = template.make_task(template.sample(Random(seed), difficulty))
    assert task == template.make_task(template.sample(Random(seed), difficulty))
    assert type(task).model_validate_json(task.model_dump_json()) == task


@PROPERTY
@given(st.sampled_from(TEMPLATES.families()), st.integers(0, 10**6), st.integers(0, 50))
def test_mutations_always_yield_valid_distinct_tasks(family: str, seed: int, variant: int) -> None:
    template = TEMPLATES.get(family)
    task = template.make_task(template.sample(Random(seed), 2))
    mutator = TaskMutator(TEMPLATES, seed=seed)
    for kind in mutator.supported(task):
        child = mutator.mutate(task, kind, variant=variant)
        if child is None:
            continue
        assert child.task_id != task.task_id and child.family == task.family
        assert child.metadata.parent_task_id == task.task_id
        assert template.make_task(template.params_of(child)).task_id == child.task_id


@PROPERTY
@given(
    st.lists(st.integers(0, 10**6), min_size=1, max_size=40, unique=True),
    st.sampled_from(["task", "family"]),
    st.text(min_size=1, max_size=8),
)
def test_split_invariants(seeds: list[int], group_by: str, salt: str) -> None:
    tasks = [make_task(FAMILIES[s % len(FAMILIES)], seed=s) for s in seeds]
    config = SplitConfig(train=0.7, validation=0.15, evaluation=0.15, group_by=group_by, salt=salt)  # type: ignore[arg-type]
    parts = split_tasks(tasks, config)
    ids = [t.task_id for split in Split for t in parts[split]]
    assert sorted(ids) == sorted(t.task_id for t in tasks)
    assert split_tasks(list(reversed(tasks)), config)[Split.TRAIN] == list(
        reversed(parts[Split.TRAIN])
    )
    if group_by == "family":
        families = {split: {t.family for t in parts[split]} for split in Split}
        assert not (families[Split.TRAIN] & families[Split.EVALUATION])
    for task in tasks:
        assert task in parts[assign_split(task, config)]


answers = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=200)
ARITH = make_task("arithmetic.integer_chain", seed=5)


@PROPERTY
@given(answers)
def test_trajectories_roundtrip_and_rewards_stay_in_bounds(text: str) -> None:
    trajectory = play(ARITH, [text], reward="rlvr_default")
    assert Trajectory.model_validate_json(trajectory.model_dump_json()) == trajectory
    assert trajectory.rewards is not None and -1.0 <= trajectory.rewards.total <= 1.1
    assert trajectory.verification is not None and 0.0 <= trajectory.verification.score <= 1.0


@PROPERTY
@given(answers, st.floats(0.05, 1.0), st.floats(0.0, 0.3), st.integers(16, 2048))
def test_composite_reward_respects_declared_bounds(
    text: str, weight: float, fmt: float, max_tokens: int
) -> None:
    trajectory = play(ARITH, [text])
    composite = CompositeReward(
        "p",
        [
            WeightedVerifierReward(weight=weight),
            BinaryReward(),
            FormatReward(weight=fmt),
            LengthPenalty(max_tokens=max_tokens, buffer_tokens=max(1, max_tokens // 4)),
            ToolCostPenalty(),
        ],
        max_bonus_ratio=1.0,
    )
    result = composite.compute(ARITH, trajectory, trajectory.verification)  # type: ignore[arg-type]
    for component, (low, high) in composite.bounds().items():
        value = result.components.as_dict()[component.value]
        assert low - 1e-12 <= value <= high + 1e-12


@PROPERTY
@given(st.integers(1, 30), st.data())
def test_pass_at_k_is_a_probability_and_monotone_in_k(n: int, data: st.DataObject) -> None:
    c = data.draw(st.integers(0, n))
    values = [pass_at_k(n, c, k) for k in range(1, n + 1)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(values))
    interval = wilson_interval(c, n)
    assert interval is not None and 0.0 <= interval[0] <= c / n <= interval[1] <= 1.0


@PROPERTY
@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(max_size=5),
        lambda inner: (
            st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=3), inner, max_size=3)
        ),
        max_leaves=10,
    )
)
def test_stable_hash_is_deterministic(value: object) -> None:
    assert stable_hash(value) == stable_hash(value)
