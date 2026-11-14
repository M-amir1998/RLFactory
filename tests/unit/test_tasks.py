from __future__ import annotations

from pathlib import Path
from random import Random

import pytest

from rlfactory.domain.task import TaskDomain
from rlfactory.errors import ConfigError, RegistryError
from rlfactory.tasks.contamination import ContaminationFilter, deduplicate, prompt_fingerprint
from rlfactory.tasks.difficulty import estimate_difficulty
from rlfactory.tasks.generators.base import MutationKind
from rlfactory.tasks.generators.catalog import default_template_registry
from rlfactory.tasks.generators.python_tasks import rename_function
from rlfactory.tasks.io import load_tasks, save_tasks
from rlfactory.tasks.mutation import TaskMutator
from rlfactory.tasks.prompts import FORMAT_INSTRUCTIONS, PromptBuilder
from rlfactory.tasks.registry import TaskRegistry, TaskSuiteConfig, materialize_suite
from rlfactory.tasks.sampling import CurriculumSampler, SamplingStrategy, TaskStats, task_stats
from rlfactory.tasks.splits import Split, SplitConfig, assign_split, split_tasks
from rlfactory.tasks.synthesis import GeneratedSource, SynthesisPipeline, generate_from_source
from tests.helpers import make_task, play

TEMPLATES = default_template_registry()


def test_catalog_covers_all_domains_with_25_families() -> None:
    assert len(TEMPLATES.families()) == 25
    assert {TEMPLATES.get(f).domain for f in TEMPLATES.families()} == set(TaskDomain)
    with pytest.raises(RegistryError):
        TEMPLATES.get("arithmetic.missing")
    with pytest.raises(RegistryError, match="unknown"):
        TEMPLATES.by_domain(TaskDomain.PYTHON, ["nope"])


@pytest.mark.parametrize("family", default_template_registry().families())
def test_every_template_is_deterministic_and_roundtrips(family: str) -> None:
    template = TEMPLATES.get(family)
    first = template.make_task(template.sample(Random(7), 3))
    second = template.make_task(template.sample(Random(7), 3))
    assert first == second
    assert first.family == family
    assert template.make_task(template.params_of(first)) == first
    assert first.metadata.difficulty is not None and 1 <= first.metadata.difficulty.level <= 5


def test_generate_from_source_is_unique_and_reproducible() -> None:
    source = GeneratedSource(domain=TaskDomain.ARITHMETIC, count=12, seed=3)
    tasks = generate_from_source(source, TEMPLATES)
    assert len({t.task_id for t in tasks}) == len(tasks) == 12
    assert [t.task_id for t in tasks] == [
        t.task_id for t in generate_from_source(source, TEMPLATES)
    ]
    with pytest.raises(ValueError, match="difficulty"):
        GeneratedSource(domain=TaskDomain.ARITHMETIC, count=1, difficulty=(4, 2))


def test_rename_function_renames_definition_and_calls() -> None:
    source = "def foo(x):\n    return foo if x else 0\n"
    renamed = rename_function(source, "foo", "bar")
    assert "def bar(x)" in renamed and "return bar" in renamed and "foo" not in renamed


def test_mutations_produce_new_valid_tasks_with_lineage() -> None:
    mutator = TaskMutator(TEMPLATES, seed=1)
    task = make_task("structured_output.logic_constraints", seed=2)
    assert MutationKind.CONSTRAINTS in mutator.supported(task)
    for kind in mutator.supported(task):
        child = mutator.mutate(task, kind)
        assert child is not None and child.task_id != task.task_id
        assert (
            child.metadata.parent_task_id == task.task_id and child.metadata.mutation == kind.value
        )
        assert child.metadata.source == "mutation"
    assert mutator.mutate(task, MutationKind.EDGE_CASE) is None
    assert len(mutator.mutate_many(task, 4)) == 4
    assert mutator.mutate_many(task, 0) == []


def test_python_variable_name_mutation_keeps_reference_valid() -> None:
    task = make_task("python.second_largest", seed=1)
    child = TaskMutator(TEMPLATES, seed=1).mutate(task, MutationKind.VARIABLE_NAMES)
    assert child is not None
    assert child.payload.entry_point != task.payload.entry_point  # type: ignore[union-attr]
    assert f"def {child.payload.entry_point}(" in child.payload.reference_solution  # type: ignore[union-attr]


def test_difficulty_levels_are_bounded_and_monotone() -> None:
    easy = estimate_difficulty({"operations": 1})
    hard = estimate_difficulty({"operations": 6, "max_operand": 1000, "tool_steps": 5})
    assert 1 <= easy.level <= hard.level <= 5


def test_prompt_builder_shared_by_all_paths() -> None:
    task = make_task("tool_use.order_total")
    messages = PromptBuilder().initial_messages(
        task,
        __import__("rlfactory.domain.trajectory", fromlist=["Observation"]).Observation(
            content="obs"
        ),
    )
    assert (
        messages[0].role.value == "system"
        and FORMAT_INSTRUCTIONS[task.domain] in messages[0].content
    )
    assert messages[1].content == "obs"
    assert set(FORMAT_INSTRUCTIONS) == set(TaskDomain)


def test_splits_are_deterministic_partitions() -> None:
    tasks = generate_from_source(
        GeneratedSource(domain=TaskDomain.PYTHON, count=40, seed=1), TEMPLATES
    )
    config = SplitConfig(train=0.6, validation=0.2, evaluation=0.2)
    parts = split_tasks(tasks, config)
    assert sum(len(v) for v in parts.values()) == len(tasks)
    assert parts == split_tasks(list(reversed(tasks)), config) or all(
        {t.task_id for t in parts[s]}
        == {t.task_id for t in split_tasks(list(reversed(tasks)), config)[s]}
        for s in Split
    )
    grouped = split_tasks(tasks, SplitConfig(group_by="family"))
    for split, members in grouped.items():
        for task in members:
            assert assign_split(task, SplitConfig(group_by="family")) is split
    families = {split: {t.family for t in members} for split, members in grouped.items()}
    assert not (families[Split.TRAIN] & families[Split.EVALUATION])
    with pytest.raises(ValueError, match="sum to 1"):
        SplitConfig(train=0.5, validation=0.1, evaluation=0.1)


def test_contamination_filter_reasons() -> None:
    evaluation = make_task("arithmetic.integer_chain", seed=1)
    flt = ContaminationFilter.from_tasks([evaluation], benchmark_ids=["gsm8k-1"])
    assert flt.reason(evaluation) == "evaluation_task_id"
    child = TaskMutator(TEMPLATES).mutate(evaluation, MutationKind.NUMBERS)
    assert child is not None and flt.reason(child) == "derived_from_evaluation_task"
    clone = evaluation.model_copy(update={"task_id": "other-id"})
    assert flt.reason(clone) == "evaluation_prompt_match"
    stranger = make_task("arithmetic.integer_chain", seed=99)
    assert flt.reason(stranger) is None
    family_filter = ContaminationFilter.from_tasks([evaluation], exclude_families=True)
    assert family_filter.reason(stranger) == "held_out_family"
    tagged = stranger.model_copy(
        update={"metadata": stranger.metadata.model_copy(update={"benchmark_id": "gsm8k-1"})}
    )
    assert flt.reason(tagged) == "benchmark_id"
    assert ContaminationFilter.from_dict(flt.to_dict()) == flt
    assert ContaminationFilter().empty and not flt.empty
    assert prompt_fingerprint(" A  b ") == prompt_fingerprint("a b")


def test_deduplicate_by_id_and_prompt() -> None:
    task = make_task("transformation.sort_lines")
    unique, duplicates = deduplicate([task, task, task.model_copy(update={"task_id": "x"})])
    assert unique == [task] and len(duplicates) == 2


def test_sampling_strategies() -> None:
    tasks = generate_from_source(
        GeneratedSource(domain=TaskDomain.ARITHMETIC, count=30, seed=5), TEMPLATES
    )
    uniform = CurriculumSampler(SamplingStrategy.UNIFORM, seed=1).sample(tasks, 10)
    assert len(uniform) == 10 and len({t.task_id for t in uniform}) == 10
    assert uniform == CurriculumSampler(SamplingStrategy.UNIFORM, seed=1).sample(tasks, 10)
    balanced = CurriculumSampler(SamplingStrategy.DIFFICULTY_BALANCED).sample(tasks, 8)
    assert len({t.metadata.difficulty.level for t in balanced if t.metadata.difficulty}) > 1
    # 1000 attempts give smoothed failure weights of ~0.999 vs ~0.001 (a ~1000:1 ratio).
    stats = {
        t.task_id: TaskStats(
            attempts=1000, passes=0 if i < 5 else 1000, mean_reward=0.0 if i < 5 else 1.0
        )
        for i, t in enumerate(tasks)
    }
    focused = CurriculumSampler(SamplingStrategy.FAILURE_FOCUSED, stats=stats).sample(tasks, 5)
    hard = {t.task_id for t in tasks[:5]}
    assert sum(t.task_id in hard for t in focused) == 5
    stratified = CurriculumSampler(SamplingStrategy.REWARD_STRATIFIED, stats=stats).sample(tasks, 6)
    assert len(stratified) == 6
    assert len(CurriculumSampler(SamplingStrategy.UNIFORM).sample(tasks, 1000)) == len(tasks)


def test_task_stats_from_trajectories() -> None:
    task = make_task("arithmetic.percentage", seed=3)
    good = play(task, [f"<answer>{task.payload.answer}</answer>"], sample_index=0)  # type: ignore[union-attr]
    bad = play(task, ["<answer>-1</answer>"], sample_index=1)
    stats = task_stats([good, bad])
    assert stats[task.task_id].attempts == 2 and stats[task.task_id].passes == 1
    assert stats[task.task_id].failure_rate == 0.5


def test_task_io_roundtrip(tmp_path: Path) -> None:
    tasks = generate_from_source(
        GeneratedSource(domain=TaskDomain.TOOL_USE, count=3, seed=2), TEMPLATES
    )
    path = tmp_path / "suite.jsonl"
    assert save_tasks(tasks, path) == 3
    assert load_tasks(path) == tasks
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_tasks(path)


def test_registry_discovers_repo_suites(config_dir: Path) -> None:
    registry = TaskRegistry(config_dir)
    assert {"train-mixed", "eval-heldout", "ci-tiny", "ci-tiny-eval"} <= set(registry.names())
    suite = registry.load("ci-tiny")
    assert suite.ref().num_tasks == len(suite) > 0
    assert registry.load("ci-tiny") is suite
    assert suite.get(suite.tasks[0].task_id) == suite.tasks[0]
    with pytest.raises(RegistryError):
        registry.load("missing")


def test_suite_verifier_overrides_and_file_sources(tmp_path: Path) -> None:
    tasks = generate_from_source(
        GeneratedSource(domain=TaskDomain.ARITHMETIC, count=2, seed=1), TEMPLATES
    )
    save_tasks(tasks, tmp_path / "fixture.jsonl")
    config = TaskSuiteConfig.model_validate(
        {
            "name": "mixed",
            "sources": [
                {"domain": "arithmetic", "count": 2, "seed": 9},
                {"kind": "file", "path": "fixture.jsonl"},
            ],
            "verifier_overrides": {"arithmetic": "exact_strict"},
        }
    )
    suite = materialize_suite(config, tmp_path)
    assert len(suite) == 4 and {t.verifier for t in suite.tasks} == {"exact_strict"}


def test_synthesis_pipeline_filters_contaminated_candidates() -> None:
    evaluation = generate_from_source(
        GeneratedSource(domain=TaskDomain.ARITHMETIC, count=2, seed=1), TEMPLATES
    )
    pipeline = SynthesisPipeline(TEMPLATES, None, ContaminationFilter.from_tasks(evaluation))
    accepted, report = pipeline.filter([*evaluation, make_task("arithmetic.percentage", seed=77)])
    assert len(accepted) == 1 and len(report.contaminated) == 2
    assert report.to_dict()["accepted"] == 1
