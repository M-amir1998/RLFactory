"""Response oracle: writes correct, wrong or malformed responses for synthetic tasks.

Used by the mock policy (CI rollouts) and by task validation (every generated task must be solvable
by its reference solution and must reject a known-wrong answer). It reads reference solutions from
task payloads, so it must never be used to produce training data that is presented as model output.
"""

from __future__ import annotations

import json
from enum import StrEnum
from fractions import Fraction
from random import Random

from pydantic import JsonValue

from rlfactory.domain.numeric import canonical_number, parse_number
from rlfactory.domain.task import (
    ArithmeticPayload,
    PythonFunctionPayload,
    StructuredOutputPayload,
    Task,
    ToolCall,
    ToolUsePayload,
    TransformationPayload,
)
from rlfactory.environments.parsing import render_tool_call


class ResponseMode(StrEnum):
    CORRECT = "correct"
    WRONG = "wrong"
    MALFORMED = "malformed"


_FILLER = (
    "Let me restate the problem to make sure I understand it.",
    "I will work through it step by step.",
    "First I identify the relevant quantities.",
    "Then I combine them carefully.",
    "I double-check the intermediate values.",
    "Everything looks consistent.",
)


def _reasoning(rng: Random, verbose: bool) -> str:
    if not verbose:
        return ""
    count = rng.randint(4, 14)
    return " ".join(_FILLER[i % len(_FILLER)] for i in range(count)) + "\n"


def _wrong_number(value: Fraction, rng: Random) -> Fraction:
    candidates = [value + 1, value - 1, value + 2, value * 2 + 1, -value - 1, value + 10]
    candidates = [c for c in candidates if c != value]
    return rng.choice(candidates)


def _equivalent_number_text(answer: str, rng: Random) -> str:
    """A different but numerically equal rendering (exercises strict vs robust verifiers)."""
    value = parse_number(answer)
    if value is None or rng.random() < 0.5:
        return answer
    if value.denominator == 1:
        return f"{value.numerator}.0"
    if "/" not in answer:
        return f"{value.numerator}/{value.denominator}"
    return answer


def _perturb_json(value: JsonValue, rng: Random) -> JsonValue:
    if isinstance(value, list):
        if len(value) > 1 and rng.random() < 0.5:
            return list(reversed(value)) if list(reversed(value)) != value else value[:-1]
        return [*value, value[0]] if value else ["unexpected"]
    if isinstance(value, dict) and value:
        key = sorted(value)[rng.randrange(len(value))]
        current = value[key]
        changed: JsonValue = (
            current + 1
            if isinstance(current, int) and not isinstance(current, bool)
            else f"{current}_x"
        )
        return {**value, key: changed}
    return value


def _buggy_python(payload: PythonFunctionPayload, rng: Random) -> str:
    renamed = payload.reference_solution.replace(
        f"def {payload.entry_point}(", "def _reference_impl(", 1
    )
    variant = rng.choice(["perturb", "perturb", "raise"])
    if variant == "raise":
        body = "    raise ValueError('unsupported input')"
    else:
        body = (
            "    result = _reference_impl(*args)\n"
            "    if isinstance(result, bool):\n        return not result\n"
            "    if isinstance(result, int):\n        return result + 1\n"
            "    if isinstance(result, list):\n        return result[:-1] if result else [0]\n"
            "    if isinstance(result, str):\n        return result + '!'\n"
            "    if isinstance(result, dict):\n        return {}\n"
            "    return 0"
        )
    return f"{renamed}\n\ndef {payload.entry_point}(*args):\n{body}\n"


class ResponseOracle:
    """Deterministic response writer. ``rng`` controls stylistic and error variation."""

    def respond(
        self,
        task: Task,
        mode: ResponseMode,
        *,
        verbose: bool,
        turn: int,
        rng: Random,
        canonical: bool = False,
    ) -> str:
        """``canonical=True`` disables equivalent-form variation (used by task validation)."""
        payload = task.payload
        if isinstance(payload, ToolUsePayload):
            return self._tool_turn(payload, mode, turn, rng)
        prefix = _reasoning(rng, verbose)
        if isinstance(payload, ArithmeticPayload):
            return prefix + self._arithmetic(payload, mode, rng, canonical)
        if isinstance(payload, StructuredOutputPayload):
            return prefix + self._structured(payload, mode, rng)
        if isinstance(payload, PythonFunctionPayload):
            return prefix + self._python(payload, mode, rng)
        return prefix + self._transformation(payload, mode)

    @staticmethod
    def _arithmetic(
        payload: ArithmeticPayload, mode: ResponseMode, rng: Random, canonical: bool
    ) -> str:
        if mode is ResponseMode.MALFORMED:
            return rng.choice(
                [f"The answer is {payload.answer}.", "<answer></answer>", "I am not sure."]
            )
        if mode is ResponseMode.CORRECT:
            text = payload.answer if canonical else _equivalent_number_text(payload.answer, rng)
            return f"<answer>{text}</answer>"
        value = parse_number(payload.answer) or Fraction(0)
        return f"<answer>{canonical_number(_wrong_number(value, rng))}</answer>"

    @staticmethod
    def _structured(payload: StructuredOutputPayload, mode: ResponseMode, rng: Random) -> str:
        truth = payload.expected if payload.expected is not None else payload.reference_answer
        text = json.dumps(truth)
        if mode is ResponseMode.MALFORMED:
            return rng.choice([f"<answer>{text[:-1]}</answer>", f"Result: {text}"])
        if mode is ResponseMode.CORRECT:
            return f"<answer>{text}</answer>"
        return f"<answer>{json.dumps(_perturb_json(truth, rng))}</answer>"

    @staticmethod
    def _python(payload: PythonFunctionPayload, mode: ResponseMode, rng: Random) -> str:
        if mode is ResponseMode.MALFORMED:
            return f"Here is the function:\n{payload.reference_solution}"
        code = (
            payload.reference_solution
            if mode is ResponseMode.CORRECT
            else _buggy_python(payload, rng)
        )
        return f"```python\n{code.rstrip()}\n```"

    @staticmethod
    def _transformation(payload: TransformationPayload, mode: ResponseMode) -> str:
        if mode is ResponseMode.MALFORMED:
            return payload.expected_output
        if mode is ResponseMode.CORRECT:
            return f"<answer>\n{payload.expected_output}\n</answer>"
        lines = payload.expected_output.split("\n")
        if len(lines) > 1 and lines[0] != lines[-1]:
            lines[0], lines[-1] = lines[-1], lines[0]
        else:
            lines[0] = lines[0].swapcase() if lines[0].swapcase() != lines[0] else lines[0] + "!"
        return "<answer>\n" + "\n".join(lines) + "\n</answer>"

    @staticmethod
    def _tool_turn(payload: ToolUsePayload, mode: ResponseMode, turn: int, rng: Random) -> str:
        plan = payload.reference_plan
        if mode is ResponseMode.MALFORMED:
            return rng.choice(
                ["Let me think about which tool to use.", "<tool_call>{bad json</tool_call>"]
            )
        if mode is ResponseMode.CORRECT:
            if turn < len(plan):
                return render_tool_call(plan[turn])
            return f"<answer>{payload.expected_answer}</answer>"
        # WRONG: sometimes misuse a tool first, then guess a wrong answer.
        if turn == 0 and rng.random() < 0.4:
            return render_tool_call(ToolCall(name="web_search", arguments={"query": "answer"}))
        if turn < min(len(plan), 2) and rng.random() < 0.5:
            return render_tool_call(plan[turn])
        value = parse_number(payload.expected_answer)
        wrong = (
            canonical_number(_wrong_number(value, rng))
            if value is not None
            else payload.expected_answer + "?"
        )
        return f"<answer>{wrong}</answer>"
