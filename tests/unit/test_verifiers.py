"""Known-value tests for every verifier: correct, incorrect, malformed and edge cases."""

from __future__ import annotations

import pytest

from rlfactory.domain.task import TaskDomain, ToolCall
from rlfactory.domain.verification import ReasonCode, VerifierKind
from rlfactory.environments.parsing import render_tool_call
from rlfactory.errors import ConfigError, RegistryError, VerifierError
from rlfactory.policies.oracle import ResponseMode
from rlfactory.sandbox.subprocess_sandbox import SubprocessSandbox
from rlfactory.verifiers.calibration import (
    binary_agreement,
    position_flip_rate,
    verbosity_flip_rate,
)
from rlfactory.verifiers.composite import CompositeComponent, CompositeVerifier
from rlfactory.verifiers.judge import LLMJudgeVerifier
from rlfactory.verifiers.matching import ExactMatchVerifier, NumericVerifier, RegexVerifier
from rlfactory.verifiers.registry import CompositeComponentSpec, CompositeSpec, VerifierRegistry
from tests.helpers import deps, make_task, oracle_play, play, tasks_for

pytestmark = pytest.mark.verifier

ARITH = make_task("arithmetic.integer_chain", seed=4)  # integer answers only


@pytest.mark.parametrize(
    ("answer", "verifier", "passed", "reason"),
    [
        ("<answer>{}</answer>", "numeric", True, ReasonCode.OK),
        ("<answer>{}.0</answer>", "numeric", True, ReasonCode.OK),
        ("<answer>{}</answer>", "exact_strict", True, ReasonCode.OK),
        ("<answer>{}.0</answer>", "exact_strict", False, ReasonCode.WRONG_ANSWER),
        ("<answer>-1</answer>", "numeric", False, ReasonCode.WRONG_ANSWER),
        ("<answer>twelve</answer>", "numeric", False, ReasonCode.INVALID_FORMAT),
        ("The answer is {}", "numeric", False, ReasonCode.NO_ANSWER),
        ("The answer is {}", "numeric_lenient", True, ReasonCode.OK),
    ],
)
def test_numeric_and_exact_known_values(
    answer: str, verifier: str, passed: bool, reason: ReasonCode
) -> None:
    value = ARITH.payload.answer  # type: ignore[union-attr]
    trajectory = play(ARITH, [answer.format(value)], verifier=verifier)
    assert trajectory.verification is not None
    assert trajectory.verification.passed is passed
    assert trajectory.verification.reason_code == reason.value
    assert 0.0 <= trajectory.verification.score <= 1.0


def test_numeric_tolerance_and_fraction_equivalence() -> None:
    task = make_task("arithmetic.fraction_sum", seed=5)
    exact = NumericVerifier("n")
    tolerant = NumericVerifier("t", rel_tol=1e-2)
    from rlfactory.domain.numeric import parse_number

    value = parse_number(task.payload.answer)  # type: ignore[union-attr]
    assert value is not None
    nearby = float(value) * 1.001
    trajectory = play(task, [f"<answer>{nearby:.12f}</answer>"])
    assert not exact.verify(task, trajectory).passed
    assert tolerant.verify(task, trajectory).passed
    with pytest.raises(ValueError, match="non-negative"):
        NumericVerifier("bad", rel_tol=-1)


def test_numeric_verifier_requires_numeric_reference() -> None:
    task = make_task("transformation.sort_lines")
    trajectory = play(task, ["<answer>x</answer>"])
    with pytest.raises(VerifierError, match="numeric reference"):
        NumericVerifier().verify(task, trajectory)


def test_exact_match_whitespace_and_case() -> None:
    task = make_task("transformation.snake_case", seed=1)
    expected = task.payload.expected_output  # type: ignore[union-attr]
    padded = play(task, [f"<answer>\n{expected}   \n\n</answer>"])
    assert padded.verification is not None and padded.verification.passed
    upper = play(task, [f"<answer>{expected.upper()}</answer>"])
    assert not ExactMatchVerifier().verify(task, upper).passed
    assert ExactMatchVerifier(case_sensitive=False).verify(task, upper).passed


def test_exact_match_needs_reference() -> None:
    task = make_task("python.count_vowels")
    trajectory = play(task, ["```python\ndef count_vowels(t):\n    return 0\n```"])
    with pytest.raises(VerifierError, match="reference"):
        ExactMatchVerifier().verify(task, trajectory)


def test_json_schema_verifier_known_values() -> None:
    task = make_task("structured_output.aggregate", seed=2)
    expected = task.payload.expected  # type: ignore[union-attr]
    import json

    cases = [
        (f"<answer>{json.dumps(expected)}</answer>", True, ReasonCode.OK),
        ('<answer>{"count": "two", "total_value": 1}</answer>', False, ReasonCode.SCHEMA_VIOLATION),
        ('<answer>{"count": 999, "total_value": 1}</answer>', False, ReasonCode.VALUE_MISMATCH),
        ("<answer>{broken</answer>", False, ReasonCode.INVALID_JSON),
        ("no answer", False, ReasonCode.NO_ANSWER),
    ]
    for text, passed, reason in cases:
        result = play(task, [text]).verification
        assert (
            result is not None and result.passed is passed and result.reason_code == reason.value
        ), text


def test_json_schema_constraints_accept_any_valid_assignment() -> None:
    task = make_task("structured_output.logic_constraints", seed=3)
    reference = oracle_play(task, ResponseMode.CORRECT)
    assert reference.passed
    names = list(task.payload.reference_answer)  # type: ignore[union-attr,arg-type]
    wrong = play(task, ["<answer>{" + ", ".join(f'"{n}": 0' for n in names) + "}</answer>"])
    assert wrong.verification is not None
    assert wrong.verification.reason_code in {
        ReasonCode.CONSTRAINT_VIOLATION.value,
        ReasonCode.OK.value,
    }


def test_bool_never_equals_int_in_json_comparison() -> None:
    from rlfactory.verifiers.base import json_equal

    assert not json_equal(True, 1)
    assert json_equal(1, 1.0)
    assert not json_equal({"a": [1, 2]}, {"a": [2, 1]})
    assert json_equal({"a": [1, {"b": None}]}, {"a": [1, {"b": None}]})


@pytest.mark.sandbox
def test_unit_test_verifier_partial_credit_and_reasons() -> None:
    d = deps(SubprocessSandbox())
    task = make_task("python.running_max", seed=2)
    good = oracle_play(task, ResponseMode.CORRECT, dependencies=d)
    assert (
        good.verification is not None
        and good.verification.passed
        and good.verification.score == 1.0
    )
    syntax = play(task, ["```python\ndef running_max(values)\n    return []\n```"], dependencies=d)
    assert (
        syntax.verification is not None
        and syntax.verification.reason_code == ReasonCode.SYNTAX_ERROR.value
    )
    no_code = play(task, ["just prose"], dependencies=d)
    assert (
        no_code.verification is not None
        and no_code.verification.reason_code == ReasonCode.NO_CODE.value
    )
    partial = play(
        task, ["```python\ndef running_max(values):\n    return list(values)\n```"], dependencies=d
    )
    result = partial.verification
    assert result is not None and not result.passed
    assert result.reason_code == ReasonCode.TESTS_FAILED.value
    assert 0.0 <= result.score < 1.0
    binary = d.verifiers.get("unit_test_binary").verify(task, partial)
    assert binary.score == 0.0


def test_regex_verifier() -> None:
    task = make_task("arithmetic.integer_chain")
    trajectory = play(task, ["thinking <answer>5</answer>"])
    assert RegexVerifier("r", r"<answer>.*?</answer>").verify(task, trajectory).passed
    assert not RegexVerifier("r", r"^\d+$", mode="fullmatch").verify(task, trajectory).passed
    assert (
        RegexVerifier("r", r"^5$", target="answer", mode="fullmatch")
        .verify(task, trajectory)
        .passed
    )
    assert RegexVerifier("r", "x").kind is VerifierKind.HEURISTIC


def test_tool_outcome_verifier() -> None:
    task = make_task("tool_use.inventory_value", seed=4)
    good = oracle_play(task, ResponseMode.CORRECT)
    assert good.verification is not None and good.verification.passed
    guessed = play(task, [f"<answer>{task.payload.expected_answer}</answer>"])  # type: ignore[union-attr]
    assert guessed.verification is not None
    assert guessed.verification.reason_code == ReasonCode.REQUIRED_TOOL_NOT_USED.value
    lenient = VerifierRegistry(
        {
            "lenient": __import__(
                "rlfactory.verifiers.registry", fromlist=["ToolOutcomeSpec"]
            ).ToolOutcomeSpec(require_tools=False)
        }
    )
    assert lenient.get("lenient").verify(task, guessed).passed
    stalled = play(task, ["hmm"])
    assert stalled.verification is not None
    assert stalled.verification.reason_code == ReasonCode.STEP_LIMIT.value
    wrong = play(
        task,
        [
            render_tool_call(ToolCall(name="kv_lookup", arguments={"key": "x"})),
            "<answer>-5</answer>",
        ],
    )
    assert (
        wrong.verification is not None
        and wrong.verification.reason_code == ReasonCode.WRONG_ANSWER.value
    )


def test_composite_verifier_weights_and_required_components() -> None:
    registry = VerifierRegistry()
    task = make_task("arithmetic.integer_chain", seed=6)
    answer = task.payload.answer  # type: ignore[union-attr]
    composite = registry.get("arithmetic_composite")
    full = composite.verify(task, play(task, [f"<answer>{answer}</answer>"]))
    assert full.passed and full.score == pytest.approx(1.0)
    assert {c.role for c in full.components} == {"correctness", "format"}
    format_only = composite.verify(task, play(task, ["<answer>-123456</answer>"]))
    assert not format_only.passed
    assert format_only.score == pytest.approx(0.1)
    assert format_only.reason_code == ReasonCode.WRONG_ANSWER.value


def test_composite_rejects_format_dominated_weights() -> None:
    registry = VerifierRegistry()
    numeric, fmt = registry.get("numeric"), registry.get("answer_tag_format")
    with pytest.raises(ConfigError, match="correctness carries"):
        CompositeVerifier(
            "bad",
            [
                CompositeComponent(numeric, "correctness", 0.2),
                CompositeComponent(fmt, "format", 0.8),
            ],
        )
    with pytest.raises(ConfigError, match="positive"):
        CompositeVerifier("bad", [CompositeComponent(numeric, "correctness", 0.0)])
    allowed = CompositeVerifier(
        "ablation",
        [CompositeComponent(numeric, "correctness", 0.2), CompositeComponent(fmt, "format", 0.8)],
        min_correctness_share=0.1,
    )
    assert allowed.kind is VerifierKind.DETERMINISTIC


def test_registry_versions_include_config_hash_and_detect_cycles() -> None:
    registry = VerifierRegistry()
    assert registry.get("numeric").version != registry.get("numeric_lenient").version
    assert registry.get("numeric").version.startswith("1.0.0+")
    cyclic = VerifierRegistry(
        {
            "a": CompositeSpec(
                components=(CompositeComponentSpec(verifier="b", role="correctness", weight=1.0),)
            ),
            "b": CompositeSpec(
                components=(CompositeComponentSpec(verifier="a", role="correctness", weight=1.0),)
            ),
        }
    )
    with pytest.raises(ConfigError, match="cycle"):
        cyclic.get("a")
    with pytest.raises(RegistryError):
        registry.get("missing")


class _FakeJudge:
    model = "fake-judge-1"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_llm_judge_verifier_records_provenance() -> None:
    task = make_task("transformation.reverse_words")
    trajectory = play(task, ["<answer>anything</answer>"])
    judge = _FakeJudge("VERDICT: PASS")
    result = LLMJudgeVerifier("judge", judge).verify(task, trajectory)
    assert result.passed and result.kind is VerifierKind.MODEL_JUDGE
    assert result.metadata["judge_model"] == "fake-judge-1"
    assert "prompt_template_hash" in result.metadata
    assert task.prompt in judge.prompts[0]
    assert (
        LLMJudgeVerifier("judge", _FakeJudge("VERDICT: FAIL")).verify(task, trajectory).reason_code
        == ReasonCode.JUDGE_REJECTED.value
    )
    assert (
        LLMJudgeVerifier("judge", _FakeJudge("I think so")).verify(task, trajectory).reason_code
        == ReasonCode.JUDGE_ERROR.value
    )
    with pytest.raises(ValueError, match="template"):
        LLMJudgeVerifier("judge", judge, template="no placeholders")


def test_judge_calibration_statistics() -> None:
    report = binary_agreement([True, True, False, False], [True, False, False, False])
    assert report.accuracy == 0.75
    assert report.precision == 0.5 and report.recall == 1.0
    assert report.cohen_kappa == pytest.approx(0.5)
    perfect = binary_agreement([True, False], [True, False])
    assert perfect.cohen_kappa == 1.0
    with pytest.raises(ValueError, match="same length"):
        binary_agreement([True], [])
    assert (
        verbosity_flip_rate(lambda text: len(text) > 20, ["short", "also short"], "x" * 30) == 1.0
    )
    assert position_flip_rate(lambda a, b: 0, [("a", "b")]) == 1.0
    assert position_flip_rate(lambda a, b: 0 if a < b else 1, [("a", "b")]) == 0.0


@pytest.mark.parametrize("domain", [d for d in TaskDomain if d is not TaskDomain.PYTHON])
def test_every_family_reference_passes_and_wrong_fails(domain: TaskDomain) -> None:
    for task in tasks_for(domain):
        assert oracle_play(task, ResponseMode.CORRECT).passed, task.family
        assert not oracle_play(task, ResponseMode.MALFORMED).passed, task.family
