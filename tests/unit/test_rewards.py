from __future__ import annotations

import pytest

from rlfactory.domain.reward import RewardComponent
from rlfactory.domain.task import ToolCall
from rlfactory.environments.parsing import render_tool_call
from rlfactory.errors import ConfigError, RegistryError
from rlfactory.policies.oracle import ResponseMode
from rlfactory.rewards.analysis import (
    RewardHackingAnalyzer,
    calibrate_rewards,
    histogram,
    repeated_ngram_fraction,
)
from rlfactory.rewards.functions import (
    BinaryReward,
    CompositeReward,
    FormatReward,
    LengthPenalty,
    ToolCostPenalty,
    WeightedVerifierReward,
)
from rlfactory.rewards.registry import CompositeRewardSpec, RewardRegistry
from tests.helpers import make_task, oracle_play, play

ARITH = make_task("arithmetic.integer_chain", seed=11)
RIGHT = f"<answer>{ARITH.payload.answer}</answer>"  # type: ignore[union-attr]


def _scored(text: str, reward: str = "binary"):  # type: ignore[no-untyped-def]
    trajectory = play(ARITH, [text], reward=reward)
    assert trajectory.verification is not None
    return trajectory


def test_binary_reward() -> None:
    good, bad = _scored(RIGHT), _scored("<answer>-999999</answer>")
    fn = BinaryReward()
    assert fn.compute(ARITH, good, good.verification).total == 1.0  # type: ignore[arg-type]
    assert fn.compute(ARITH, bad, bad.verification).total == 0.0  # type: ignore[arg-type]
    assert BinaryReward(success=1, failure=-1).bounds()[RewardComponent.CORRECTNESS] == (-1, 1)


def test_weighted_verifier_reward_and_validation() -> None:
    good = _scored(RIGHT)
    assert WeightedVerifierReward(weight=0.5).compute(ARITH, good, good.verification).total == 0.5  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        WeightedVerifierReward(weight=0)


def test_format_reward() -> None:
    fn = FormatReward(weight=0.1, penalty=0.2)
    formatted, unformatted = _scored("<answer>1</answer>"), _scored("no tags")
    assert fn.compute(ARITH, formatted, formatted.verification).components.format == 0.1  # type: ignore[arg-type]
    assert fn.compute(ARITH, unformatted, unformatted.verification).components.format == -0.2  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        FormatReward(weight=-1)


@pytest.mark.parametrize(
    ("length", "expected"), [(0, 0.0), (768, 0.0), (896, -0.25), (1024, -0.5), (5000, -0.5)]
)
def test_length_penalty_soft_overlong(length: int, expected: float) -> None:
    assert LengthPenalty(max_tokens=1024, buffer_tokens=256, coefficient=0.5).penalty(
        length
    ) == pytest.approx(expected)


def test_length_penalty_validation() -> None:
    with pytest.raises(ConfigError):
        LengthPenalty(max_tokens=10, buffer_tokens=20)


def test_tool_cost_penalty_only_beyond_plan() -> None:
    task = make_task("tool_use.order_total", seed=2)
    fn = ToolCostPenalty(cost_per_call=0.1, error_cost=0.05, slack_calls=0, max_penalty=0.3)
    efficient = oracle_play(task, ResponseMode.CORRECT)
    assert fn.compute(task, efficient, efficient.verification).components.tool_cost == 0.0  # type: ignore[arg-type]
    plan = task.payload.reference_plan  # type: ignore[union-attr]
    wasteful = play(
        task,
        [
            render_tool_call(plan[0]),
            render_tool_call(ToolCall(name="nope")),
            render_tool_call(plan[0]),
            f"<answer>{task.payload.expected_answer}</answer>",
        ],
    )  # type: ignore[union-attr]
    penalty = fn.compute(task, wasteful, wasteful.verification).components.tool_cost  # type: ignore[arg-type]
    assert -0.3 <= penalty < 0


def test_composite_reward_sums_and_clips() -> None:
    composite = CompositeReward(
        "c",
        [BinaryReward(), FormatReward(weight=0.1), LengthPenalty(max_tokens=4, buffer_tokens=2)],
        clip_min=-1.0,
        clip_max=1.05,
    )
    good = _scored(RIGHT)
    result = composite.compute(ARITH, good, good.verification)  # type: ignore[arg-type]
    assert result.components.correctness == 1.0 and result.components.format == 0.1
    assert result.total <= 1.05
    assert result.clipped == (result.total != result.components.total())
    assert set(result.details) == {"binary", "format", "length_penalty"}


def test_composite_reward_overshaping_guard() -> None:
    with pytest.raises(ConfigError, match="over-shaped"):
        CompositeReward("bad", [WeightedVerifierReward(weight=0.4), FormatReward(weight=0.6)])
    with pytest.raises(ConfigError, match="no positive correctness"):
        CompositeReward("bad", [FormatReward(weight=0.1)])
    allowed = CompositeReward(
        "ok", [WeightedVerifierReward(weight=0.4), FormatReward(weight=0.6)], allow_overshaping=True
    )
    assert allowed.overshaped
    with pytest.raises(ConfigError, match="clip_min"):
        CompositeReward("bad", [BinaryReward()], clip_min=1, clip_max=0)


def test_reward_registry_defaults_and_overshaped_study_variant() -> None:
    registry = RewardRegistry()
    assert registry.get("binary").version != registry.get("correctness_format").version
    format_only = _scored("<answer>-424242</answer>", reward="overshaped")
    assert format_only.rewards is not None and format_only.rewards.total == pytest.approx(0.6)
    with pytest.raises(RegistryError):
        registry.get("missing")
    with pytest.raises(ConfigError, match="cycle"):
        RewardRegistry({"x": CompositeRewardSpec(components=("x",))}).get("x")


def test_hacking_analyzer_flags_overshaped_format_exploitation() -> None:
    overshaped = [
        _scored("<answer></answer>x", reward="overshaped"),
        _scored("<answer>-1</answer>", reward="overshaped"),
    ]
    report = RewardHackingAnalyzer(high_reward_threshold=0.5).analyze(overshaped)
    assert report.examined == 2
    assert "high_reward_unverified" in report.counts
    assert "Heuristic" in report.caveat
    multiple = _scored("<answer>1</answer><answer>2</answer>")
    assert "multiple_answers" in RewardHackingAnalyzer().analyze([multiple]).counts


def test_hacking_analyzer_repeated_outputs_across_tasks() -> None:
    tasks = [make_task("arithmetic.percentage", seed=s) for s in range(4)]
    same = [play(t, ["<answer>7</answer>"]) for t in tasks]
    report = RewardHackingAnalyzer(min_tasks_for_repeat=3).analyze(same)
    assert report.counts.get("repeated_output", 0) == sum(
        1 for t in same if (t.reward_total or 0) > 0
    )
    assert repeated_ngram_fraction("a b c a b c a b c a b c") > 0.5
    assert repeated_ngram_fraction("one two three") == 0.0


def test_reward_calibration_known_values() -> None:
    trajectories = [_scored(RIGHT), _scored(RIGHT), _scored("<answer>-1</answer>"), _scored("bad")]
    calibration = calibrate_rewards(trajectories)
    assert calibration.n == 4 and calibration.pass_rate == 0.5
    assert calibration.point_biserial == pytest.approx(1.0)
    assert calibration.auc == 1.0
    assert calibration.mean_reward_passed == 1.0 and calibration.mean_reward_failed == 0.0
    assert histogram([0.0, 1.0, 1.0], bins=2) == ((0.0, 0.5, 1), (0.5, 1.0, 2))
    assert histogram([]) == ()
