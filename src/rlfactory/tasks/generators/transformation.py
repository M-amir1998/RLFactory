"""Text-transformation templates: one template per deterministic string operation."""

from __future__ import annotations

from random import Random
from typing import Literal, get_args

from pydantic import Field
from typing_extensions import override

from rlfactory.domain.task import TaskDomain, TransformationPayload
from rlfactory.tasks.generators.base import BuiltTask, MutationKind, TaskTemplate, TemplateParams
from rlfactory.tasks.generators.vocabulary import WORDS

Operation = Literal[
    "reverse_words", "sort_lines", "dedupe_lines", "caesar_shift", "snake_case", "uppercase_vowels"
]

_INSTRUCTIONS: dict[str, str] = {
    "reverse_words": "Reverse the order of the words on each line, keeping the lines in order.",
    "sort_lines": "Sort the lines alphabetically, ignoring case (ties keep their original order).",
    "dedupe_lines": "Remove duplicate lines, keeping the first occurrence of each line in order.",
    "caesar_shift": (
        "Apply a Caesar shift of {shift} to every letter (shift 1 maps a to b and z to a), "
        "preserving case and leaving other characters unchanged."
    ),
    "snake_case": "Convert each line to snake_case: lowercase every word and join words with '_'.",
    "uppercase_vowels": "Convert every vowel (a, e, i, o, u) to uppercase; leave everything else as is.",
}


def _caesar(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def apply_operation(operation: Operation, lines: tuple[str, ...], shift: int) -> list[str]:
    if operation == "reverse_words":
        return [" ".join(reversed(line.split())) for line in lines]
    if operation == "sort_lines":
        return sorted(lines, key=str.lower)
    if operation == "dedupe_lines":
        return list(dict.fromkeys(lines))
    if operation == "caesar_shift":
        return [_caesar(line, shift) for line in lines]
    if operation == "snake_case":
        return ["_".join(word.lower() for word in line.split()) for line in lines]
    return ["".join(ch.upper() if ch in "aeiou" else ch for ch in line) for line in lines]


class TransformParams(TemplateParams):
    lines: tuple[str, ...] = Field(min_length=1)
    shift: int = Field(default=0, ge=0, le=25)


class TextTransformTemplate(TaskTemplate[TransformParams]):
    domain = TaskDomain.TRANSFORMATION
    environment = "transformation"
    verifier = "exact_match"
    params_model = TransformParams
    mutations = frozenset(
        {
            MutationKind.INPUT_LENGTH,
            MutationKind.NUMBERS,
            MutationKind.EDGE_CASE,
            MutationKind.FORMAT,
        }
    )

    def __init__(self, operation: Operation) -> None:
        self.operation: Operation = operation
        self.name = operation

    def _line(self, rng: Random, width: int) -> str:
        words = [rng.choice(WORDS) for _ in range(rng.randint(2, max(2, width)))]
        if self.operation == "snake_case":
            words = [w.capitalize() for w in words]
        return " ".join(words)

    def _lines(self, rng: Random, count: int, width: int) -> tuple[str, ...]:
        lines = [self._line(rng, width) for _ in range(count)]
        if self.operation == "dedupe_lines" and len(lines) > 1:
            for _ in range(max(1, count // 3)):
                lines.insert(rng.randrange(len(lines) + 1), rng.choice(lines))
        return tuple(lines)

    @override
    def sample(self, rng: Random, difficulty: int) -> TransformParams:
        shift = rng.randint(1, 25) if self.operation == "caesar_shift" else 0
        return TransformParams(lines=self._lines(rng, 2 + difficulty, 2 + difficulty), shift=shift)

    @override
    def build(self, params: TransformParams) -> BuiltTask:
        instruction = _INSTRUCTIONS[self.operation].format(shift=params.shift)
        text = "\n".join(params.lines)
        expected = "\n".join(apply_operation(self.operation, params.lines, params.shift))
        return BuiltTask(
            prompt=f"{instruction}\n\nInput:\n{text}",
            payload=TransformationPayload(
                operation=self.operation, input_text=text, expected_output=expected
            ),
            signals={
                "operations": 1.0,
                "input_size": float(sum(len(line.split()) for line in params.lines)),
            },
        )

    @override
    def mutate(
        self, params: TransformParams, kind: MutationKind, rng: Random
    ) -> TransformParams | None:
        if kind is MutationKind.INPUT_LENGTH:
            extra = self._lines(rng, 2, 4)
            return TransformParams(lines=(*params.lines, *extra), shift=params.shift)
        if kind is MutationKind.NUMBERS:
            if self.operation == "caesar_shift":
                return TransformParams(lines=params.lines, shift=rng.randint(1, 25))
            return TransformParams(lines=self._lines(rng, len(params.lines), 4), shift=params.shift)
        if kind is MutationKind.EDGE_CASE:
            single = params.lines[:1]
            if self.operation == "dedupe_lines":
                return TransformParams(lines=single * 3, shift=params.shift)
            return TransformParams(lines=single, shift=params.shift)
        if kind is MutationKind.FORMAT:
            mixed = tuple(
                line.swapcase() if i % 2 == 0 else line for i, line in enumerate(params.lines)
            )
            if mixed == params.lines:
                return None
            return TransformParams(lines=mixed, shift=params.shift)
        return None


def transformation_templates() -> list[TextTransformTemplate]:
    return [TextTransformTemplate(operation) for operation in get_args(Operation)]
