"""Python function templates: one template per reference function, hidden tests generated from a seed."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from typing import Any

from pydantic import Field, JsonValue
from typing_extensions import override

from rlfactory.domain.task import PythonFunctionPayload, PythonTestCase, TaskDomain
from rlfactory.tasks.generators import python_reference as ref
from rlfactory.tasks.generators.base import BuiltTask, MutationKind, TaskTemplate, TemplateParams
from rlfactory.tasks.generators.vocabulary import WORDS

ArgsMaker = Callable[[Random, int], tuple[JsonValue, ...]]


def rename_function(source: str, old: str, new: str) -> str:
    """Rename a function definition (and references to it) via the AST."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == old:
            node.name = new
        elif isinstance(node, ast.Name) and node.id == old:
            node.id = new
    return ast.unparse(tree) + "\n"


def _to_json(value: Any) -> JsonValue:
    loaded: JsonValue = json.loads(json.dumps(value))
    return loaded


def _ints(low: int, high: int) -> ArgsMaker:
    def make(rng: Random, size: int) -> tuple[JsonValue, ...]:
        values: list[JsonValue] = [rng.randint(low, high) for _ in range(size)]
        return (values,)

    return make


def _dupes(rng: Random, size: int) -> tuple[JsonValue, ...]:
    values: list[JsonValue] = [rng.randint(0, max(1, size // 2)) for _ in range(size)]
    return (values,)


def _sentence(rng: Random, size: int) -> tuple[JsonValue, ...]:
    words = [rng.choice(WORDS) for _ in range(max(1, size))]
    words = [w.capitalize() if rng.random() < 0.3 else w for w in words]
    return (" ".join(words),)


def _maybe_palindrome(rng: Random, size: int) -> tuple[JsonValue, ...]:
    letters = "".join(rng.choice("abcdefgh") for _ in range(max(1, size)))
    if rng.random() < 0.5:
        text = letters + letters[::-1]
        return ("".join(ch.upper() if rng.random() < 0.3 else ch for ch in text),)
    return (letters + rng.choice("xyz"),)


def _rotation(rng: Random, size: int) -> tuple[JsonValue, ...]:
    values: list[JsonValue] = [rng.randint(-9, 9) for _ in range(size)]
    return (values, rng.randint(0, size + 3))


@dataclass(frozen=True)
class FunctionSpec:
    reference: Callable[..., Any]
    signature: str
    description: str
    make_args: ArgsMaker
    edge_cases: tuple[tuple[JsonValue, ...], ...]
    alternative_names: tuple[str, ...]

    @property
    def key(self) -> str:
        return self.reference.__name__


FUNCTION_SPECS: tuple[FunctionSpec, ...] = (
    FunctionSpec(
        ref.sum_of_evens,
        "def {name}(values: list[int]) -> int",
        "returns the sum of the even numbers in `values`.",
        _ints(-20, 20),
        (([],), ([1, 3, 5],), ([-2, -4],)),
        ("total_even", "even_sum", "add_evens"),
    ),
    FunctionSpec(
        ref.count_vowels,
        "def {name}(text: str) -> int",
        "returns how many characters of `text` are vowels (a, e, i, o, u), ignoring case.",
        _sentence,
        (("",), ("xyz",), ("AEIOU",)),
        ("vowel_count", "num_vowels", "tally_vowels"),
    ),
    FunctionSpec(
        ref.running_max,
        "def {name}(values: list[int]) -> list[int]",
        "returns a list where element i is the maximum of `values[0..i]`.",
        _ints(-30, 30),
        (([],), ([5],), ([3, 3, 3],)),
        ("prefix_max", "cumulative_max", "max_so_far"),
    ),
    FunctionSpec(
        ref.dedupe_preserve_order,
        "def {name}(values: list[int]) -> list[int]",
        "returns `values` without duplicates, keeping the first occurrence of each value in order.",
        _dupes,
        (([],), ([1, 1, 1],), ([2, 1, 2, 1],)),
        ("unique_in_order", "remove_duplicates", "stable_unique"),
    ),
    FunctionSpec(
        ref.second_largest,
        "def {name}(values: list[int]) -> int | None",
        "returns the second largest distinct value in `values`, or None if there is none.",
        _ints(-15, 15),
        (([],), ([7],), ([4, 4],)),
        ("runner_up", "second_max", "next_largest"),
    ),
    FunctionSpec(
        ref.is_palindrome,
        "def {name}(text: str) -> bool",
        "returns True if `text` reads the same backwards, considering only letters and digits "
        "and ignoring case.",
        _maybe_palindrome,
        (("",), ("A man, a plan, a canal: Panama",), ("ab",)),
        ("palindrome_check", "reads_same_backwards", "check_palindrome"),
    ),
    FunctionSpec(
        ref.rotate_left,
        "def {name}(values: list[int], k: int) -> list[int]",
        "returns `values` rotated left by `k` positions (k may exceed the length).",
        _rotation,
        (([], 3), ([1, 2, 3], 0), ([1, 2, 3], 5)),
        ("shift_left", "left_rotation", "rotate_values"),
    ),
    FunctionSpec(
        ref.char_frequency,
        "def {name}(text: str) -> dict[str, int]",
        "returns a dict mapping each letter of `text` (lowercased) to its number of occurrences, "
        "ignoring non-letters.",
        _sentence,
        (("",), ("AaA!",), ("xyz xyz",)),
        ("letter_counts", "count_letters", "letter_histogram"),
    ),
)


class PythonParams(TemplateParams):
    entry_point: str
    num_tests: int = Field(default=6, ge=1, le=64)
    input_size: int = Field(default=5, ge=0, le=128)
    seed: int = Field(ge=0)
    include_edge_cases: bool = False


class PythonFunctionTemplate(TaskTemplate[PythonParams]):
    domain = TaskDomain.PYTHON
    environment = "python"
    verifier = "unit_test"
    params_model = PythonParams
    mutations = frozenset(
        {
            MutationKind.VARIABLE_NAMES,
            MutationKind.INPUT_LENGTH,
            MutationKind.EDGE_CASE,
            MutationKind.NUMBERS,
        }
    )

    def __init__(self, spec: FunctionSpec) -> None:
        self.spec = spec
        self.name = spec.key
        self._source = inspect.getsource(spec.reference)

    @override
    def sample(self, rng: Random, difficulty: int) -> PythonParams:
        return PythonParams(
            entry_point=self.spec.key,
            num_tests=4 + difficulty,
            input_size=2 + 2 * difficulty,
            seed=rng.randint(0, 10**6),
            include_edge_cases=difficulty >= 4,
        )

    @override
    def build(self, params: PythonParams) -> BuiltTask:
        rng = Random(params.seed)
        args_list = [self.spec.make_args(rng, params.input_size) for _ in range(params.num_tests)]
        if params.include_edge_cases:
            args_list.extend(self.spec.edge_cases)
        tests = tuple(
            PythonTestCase(
                args=tuple(args), expected=_to_json(self.spec.reference(*copy.deepcopy(args)))
            )
            for args in args_list
        )
        source = rename_function(self._source, self.spec.key, params.entry_point)
        signature = self.spec.signature.format(name=params.entry_point)
        examples = tests[:2]
        rendered = "\n".join(
            f"- {params.entry_point}({', '.join(repr(a) for a in case.args)}) returns {case.expected!r}"
            for case in examples
        )
        prompt = f"Write a Python function `{signature}` that {self.spec.description}\n\nExamples:\n{rendered}"
        complexity = sum(1 for _ in ast.walk(ast.parse(source)))
        return BuiltTask(
            prompt=prompt,
            payload=PythonFunctionPayload(
                entry_point=params.entry_point,
                signature=signature,
                tests=tests,
                public_examples=examples,
                reference_solution=source,
            ),
            signals={
                "tests": float(len(tests)),
                "input_size": float(params.input_size),
                "program_complexity": float(complexity),
            },
        )

    @override
    def mutate(self, params: PythonParams, kind: MutationKind, rng: Random) -> PythonParams | None:
        data = params.model_dump()
        if kind is MutationKind.VARIABLE_NAMES:
            names = [
                n for n in (self.spec.key, *self.spec.alternative_names) if n != params.entry_point
            ]
            data["entry_point"] = rng.choice(names)
        elif kind is MutationKind.INPUT_LENGTH:
            data["input_size"] = min(128, max(2, params.input_size * 2))
        elif kind is MutationKind.EDGE_CASE:
            if params.include_edge_cases:
                return None
            data["include_edge_cases"] = True
        elif kind is MutationKind.NUMBERS:
            data["seed"] = rng.randint(0, 10**6)
        else:
            return None
        return PythonParams.model_validate(data)


def python_templates() -> list[PythonFunctionTemplate]:
    return [PythonFunctionTemplate(spec) for spec in FUNCTION_SPECS]
