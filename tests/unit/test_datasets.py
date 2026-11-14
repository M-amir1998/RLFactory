from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlfactory.datasets.adapters import (
    to_openrlhf_rl,
    to_trl_dpo,
    to_trl_grpo,
    to_trl_reward,
    to_trl_sft,
    to_verl_rl,
)
from rlfactory.datasets.builders import (
    PreferenceBuilderConfig,
    PreferenceDatasetBuilder,
    PreferenceStrategy,
    RLBuilderConfig,
    RLPromptDatasetBuilder,
    SFTBuilderConfig,
    SFTDatasetBuilder,
    SFTSelection,
    jaccard,
)
from rlfactory.datasets.io import read_card, read_rows, write_dataset
from rlfactory.datasets.records import DatasetMessage
from rlfactory.datasets.reward_model import RewardModelDatasetBuilder, RewardModelEvaluator
from rlfactory.datasets.validation import (
    validate_preference,
    validate_rl,
    validate_sft,
    validate_trl_file,
)
from rlfactory.domain.common import CodeVersion
from rlfactory.domain.task import TaskDomain
from rlfactory.environments.registry import default_environment_registry
from rlfactory.policies.oracle import ResponseMode
from rlfactory.tasks.contamination import ContaminationFilter
from rlfactory.tasks.sampling import TaskStats
from tests.helpers import make_task, oracle_play, play, tasks_for

TASK = make_task("arithmetic.integer_chain", seed=21)
RIGHT = f"<answer>{TASK.payload.answer}</answer>"  # type: ignore[union-attr]
VERBOSE_RIGHT = "Let me think step by step about this carefully and slowly. " * 5 + RIGHT


def _group():  # type: ignore[no-untyped-def]
    return [
        play(TASK, [RIGHT], sample_index=0),
        play(TASK, [VERBOSE_RIGHT], sample_index=1),
        play(
            TASK,
            ["Long rambling wrong answer with many many many words here. <answer>-1</answer>"],
            sample_index=2,
        ),
        play(TASK, ["<answer>-2</answer>"], sample_index=3),
    ]


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (SFTSelection.SHORTEST_CORRECT, [RIGHT]),
        (SFTSelection.HIGHEST_REWARD, [RIGHT]),
        (SFTSelection.ALL_SUCCESSFUL, [RIGHT, VERBOSE_RIGHT]),
    ],
)
def test_sft_selection_policies(selection: SFTSelection, expected: list[str]) -> None:
    result = SFTDatasetBuilder(SFTBuilderConfig(selection=selection, max_per_task=5)).build(
        _group()
    )
    assert sorted(r.completion[0].content for r in result.records) == sorted(expected)
    assert result.excluded["not_passed"] == 2
    record = result.records[0]
    assert record.prompt[-1].role == "user" and record.completion[0].role == "assistant"
    assert record.lineage.task_id == TASK.task_id and record.lineage.verifier == "numeric"


def test_sft_diverse_selection_skips_near_duplicates() -> None:
    near_duplicate = play(
        TASK,
        ["Let me think step by step about this carefully and slowly. " * 5 + " " + RIGHT],
        sample_index=4,
    )
    config = SFTBuilderConfig(
        selection=SFTSelection.DIVERSE_CORRECT, max_per_task=5, diversity_threshold=0.8
    )
    result = SFTDatasetBuilder(config).build([*_group(), near_duplicate])
    assert len(result.records) == 2
    assert jaccard("a b c d", "a b c d") == 1.0 and jaccard("a b c", "x y z") == 0.0


def test_sft_multi_turn_expansion() -> None:
    task = make_task("tool_use.order_total", seed=5)
    trajectory = oracle_play(task, ResponseMode.CORRECT)
    expanded = SFTDatasetBuilder(SFTBuilderConfig(expand_multi_turn=True)).build([trajectory])
    last_only = SFTDatasetBuilder(SFTBuilderConfig(expand_multi_turn=False)).build([trajectory])
    assert len(expanded.records) == len(trajectory.actions) > 1 and len(last_only.records) == 1
    assert any(m.content.startswith("Tool result:") for m in expanded.records[-1].prompt)


def test_sft_excludes_contaminated_tasks() -> None:
    result = SFTDatasetBuilder(
        contamination=ContaminationFilter.from_tasks([TASK]), tasks={TASK.task_id: TASK}
    ).build(_group())
    assert result.records == [] and result.excluded["contamination:evaluation_task_id"] == 4


@pytest.mark.parametrize(
    ("strategy", "check"),
    [
        (
            PreferenceStrategy.CORRECT_VS_INCORRECT,
            lambda r: r.chosen_passed and not r.rejected_passed,
        ),
        (
            PreferenceStrategy.REWARD_MARGIN,
            lambda r: (r.chosen_reward or 0) - (r.rejected_reward or 0) >= 0.5,
        ),
        (
            PreferenceStrategy.SHORT_CORRECT_VS_VERBOSE_INCORRECT,
            lambda r: len(r.chosen[0].content) < len(r.rejected[0].content) and r.chosen_passed,
        ),
    ],
)
def test_preference_strategies(strategy: PreferenceStrategy, check) -> None:  # type: ignore[no-untyped-def]
    result = PreferenceDatasetBuilder(
        PreferenceBuilderConfig(strategy=strategy, max_pairs_per_task=3)
    ).build(_group())
    assert result.records
    for record in result.records:
        assert check(record)
        assert record.chosen != record.rejected and record.strategy == strategy.value
    assert validate_preference(result.records).ok


def test_preference_pairs_never_cross_groups() -> None:
    other_task = make_task("arithmetic.integer_chain", seed=22)
    correct_elsewhere = play(
        other_task, [f"<answer>{other_task.payload.answer}</answer>"], group_id="other-group"
    )  # type: ignore[union-attr]
    wrong_here = play(TASK, ["<answer>-5</answer>"], group_id="group-test")
    result = PreferenceDatasetBuilder().build([correct_elsewhere, wrong_here])
    assert result.records == [] and result.excluded["no_valid_pair_in_group"] == 2


def test_preference_excludes_multi_turn_groups() -> None:
    task = make_task("tool_use.order_total", seed=6)
    result = PreferenceDatasetBuilder().build(
        [oracle_play(task, ResponseMode.CORRECT), oracle_play(task, ResponseMode.MALFORMED)]
    )
    assert result.records == [] and "multi_turn" in result.excluded


def test_grpo_builder_drops_saturated_multistep_and_contaminated() -> None:
    tasks = tasks_for(TaskDomain.ARITHMETIC) + tasks_for(TaskDomain.TOOL_USE)
    stats = {
        tasks[0].task_id: TaskStats(attempts=4, passes=4, mean_reward=1.0),
        tasks[1].task_id: TaskStats(attempts=4, passes=0, mean_reward=0.0),
        tasks[2].task_id: TaskStats(attempts=4, passes=2, mean_reward=0.5),
    }
    builder = RLPromptDatasetBuilder(
        environments=default_environment_registry(),
        contamination=ContaminationFilter.from_tasks([tasks[3]]),
    )
    result = builder.build(tasks, stats)
    kept = {r.task.task_id for r in result.records}
    assert kept == {tasks[2].task_id}
    assert result.excluded == {
        "saturated_all_pass": 1,
        "saturated_all_fail": 1,
        "contamination:evaluation_task_id": 1,
        "multi_step_task": len(tasks_for(TaskDomain.TOOL_USE)),
    }
    record = result.records[0]
    assert record.pass_rate == 0.5 and record.prompt[0].role == "system"
    assert validate_rl(result.records).ok
    unfiltered = RLPromptDatasetBuilder(
        RLBuilderConfig(drop_saturated=False, max_prompt_chars=10),
        environments=default_environment_registry(),
    ).build(tasks[:2])
    assert unfiltered.excluded.get("prompt_too_long") == 2


def test_trl_adapters_match_verified_formats() -> None:
    sft = SFTDatasetBuilder().build(_group()).records[0]
    assert set(to_trl_sft(sft)) == {"prompt", "completion"}
    assert to_trl_sft(sft)["completion"] == [{"role": "assistant", "content": RIGHT}]
    pair = PreferenceDatasetBuilder().build(_group()).records[0]
    assert set(to_trl_dpo(pair)) == {"prompt", "chosen", "rejected"}
    reward_row = to_trl_reward(pair)
    assert (
        set(reward_row) == {"chosen", "rejected"}
        and reward_row["chosen"][-1]["role"] == "assistant"
    )
    rl = (
        RLPromptDatasetBuilder(environments=default_environment_registry()).build([TASK]).records[0]
    )
    grpo = to_trl_grpo(rl)
    assert set(grpo) == {"prompt", "rlfactory_task", "rlfactory_reward"}
    assert json.loads(grpo["rlfactory_task"])["task_id"] == TASK.task_id
    verl = to_verl_rl(rl, 3)
    assert set(verl) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert verl["reward_model"]["style"] == "rule" and verl["extra_info"]["index"] == 3
    assert set(to_openrlhf_rl(rl)) == {"input", "label"}


def test_validation_detects_common_defects() -> None:
    sft = SFTDatasetBuilder().build(_group()).records[0]
    empty = sft.model_copy(update={"completion": (DatasetMessage(role="assistant", content="  "),)})
    bad_prompt = sft.model_copy(update={"prompt": (DatasetMessage(role="assistant", content="x"),)})
    report = validate_sft([empty, bad_prompt, sft, sft], ContaminationFilter.from_tasks([TASK]))
    assert any("empty completion" in e for e in report.errors)
    assert any("no user message" in e for e in report.errors)
    assert any("leaked" in e for e in report.errors)
    assert report.duplicate_prompts >= 1 and not report.ok
    pair = PreferenceDatasetBuilder().build(_group()).records[0]
    same = pair.model_copy(update={"rejected": pair.chosen, "chosen_reward": float("nan")})
    preference_report = validate_preference([same])
    assert any("chosen == rejected" in e for e in preference_report.errors)
    assert any("non-finite" in e for e in preference_report.errors)


def test_write_dataset_and_validate_trl_file(tmp_path: Path) -> None:
    result = SFTDatasetBuilder(
        SFTBuilderConfig(selection=SFTSelection.ALL_SUCCESSFUL, max_per_task=5)
    ).build(_group())
    rows = [to_trl_sft(r) for r in result.records]
    card = write_dataset(
        tmp_path / "sft",
        name="sft",
        kind="sft",
        records=result.records,
        rows=rows,
        builder="SFTDatasetBuilder",
        builder_config=SFTBuilderConfig(),
        build_report=result.report(),
        validation={"ok": True},
        source_run_ids=["r", "r"],
        contamination_counts={},
        code=CodeVersion(rlfactory_version="t"),
    )
    assert (
        card.schema_name == "trl.sft.conversational.prompt_completion"
        and card.source_run_ids == ("r",)
    )
    assert (
        read_card(tmp_path / "sft") == card and read_rows(tmp_path / "sft" / "train.jsonl") == rows
    )
    assert card.reference(tmp_path / "sft").num_records == len(rows)
    assert validate_trl_file(tmp_path / "sft" / "train.jsonl", "sft").ok
    broken = tmp_path / "broken.jsonl"
    broken.write_text(
        '{"prompt": [{"role": "robot", "content": "x"}]}\nnot json\n', encoding="utf-8"
    )
    report = validate_trl_file(broken, "sft")
    assert not report.ok and report.records == 2
    assert not validate_trl_file(broken, "dpo").ok


def test_reward_model_builder_and_evaluator() -> None:
    pairs = (
        PreferenceDatasetBuilder(
            PreferenceBuilderConfig(strategy=PreferenceStrategy.REWARD_MARGIN, max_pairs_per_task=4)
        )
        .build(_group())
        .records
    )
    verified = RewardModelDatasetBuilder().build(pairs)
    assert all(p.chosen_passed and not p.rejected_passed for p in verified)
    assert RewardModelDatasetBuilder.to_rows(verified)[0].keys() == {"chosen", "rejected"}
    evaluation = RewardModelEvaluator().evaluate(
        lambda prompt, completion: -len(completion[0].content), verified
    )
    assert 0.0 <= evaluation.accuracy <= 1.0 and evaluation.pairs == len(verified)
    with pytest.raises(ValueError, match="at least one"):
        RewardModelEvaluator().evaluate(lambda p, c: 0.0, [])
