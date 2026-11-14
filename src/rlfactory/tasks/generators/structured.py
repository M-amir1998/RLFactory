"""Structured-output templates: JSON transformations, aggregation, extraction and logic puzzles."""

from __future__ import annotations

import itertools
import json
from fractions import Fraction
from random import Random
from typing import ClassVar, Literal

from pydantic import Field, JsonValue
from typing_extensions import override

from rlfactory.domain.numeric import evaluate_condition
from rlfactory.domain.task import StructuredOutputPayload, TaskDomain
from rlfactory.tasks.generators.base import BuiltTask, MutationKind, TaskTemplate, TemplateParams
from rlfactory.tasks.generators.vocabulary import CITIES, PEOPLE, PRODUCTS, VARIABLE_SETS


class Product(TemplateParams):
    name: str
    price: int = Field(ge=0)
    qty: int = Field(ge=0)


def _products(rng: Random, count: int) -> tuple[Product, ...]:
    names = rng.sample(PRODUCTS, k=min(count, len(PRODUCTS)))
    return tuple(Product(name=n, price=rng.randint(1, 50), qty=rng.randint(0, 20)) for n in names)


def _products_json(products: tuple[Product, ...]) -> str:
    return json.dumps([p.model_dump() for p in products])


class FilterSortParams(TemplateParams):
    products: tuple[Product, ...] = Field(min_length=1)
    threshold: int
    descending: bool = False


class FilterSortTemplate(TaskTemplate[FilterSortParams]):
    name = "filter_sort"
    domain = TaskDomain.STRUCTURED_OUTPUT
    environment = "structured_output"
    verifier = "json_schema"
    params_model = FilterSortParams
    mutations = frozenset(
        {
            MutationKind.NUMBERS,
            MutationKind.INPUT_LENGTH,
            MutationKind.EDGE_CASE,
            MutationKind.FORMAT,
        }
    )

    @override
    def sample(self, rng: Random, difficulty: int) -> FilterSortParams:
        return FilterSortParams(
            products=_products(rng, 3 + difficulty),
            threshold=rng.randint(5, 40),
            descending=difficulty >= 3 and rng.random() < 0.5,
        )

    @override
    def build(self, params: FilterSortParams) -> BuiltTask:
        selected = [p for p in params.products if p.price >= params.threshold]
        selected.sort(key=lambda p: (-p.price, p.name) if params.descending else (p.price, p.name))
        order = "descending" if params.descending else "ascending"
        prompt = (
            f"Here is a JSON list of products:\n{_products_json(params.products)}\n\n"
            f"Return a JSON array with the names of the products whose price is at least "
            f"{params.threshold}, sorted by price {order} (break ties by name, ascending)."
        )
        expected: list[JsonValue] = [p.name for p in selected]
        return BuiltTask(
            prompt=prompt,
            payload=StructuredOutputPayload(
                json_schema={"type": "array", "items": {"type": "string"}}, expected=expected
            ),
            signals={"operations": 2.0, "input_size": float(len(params.products))},
        )

    @override
    def mutate(
        self, params: FilterSortParams, kind: MutationKind, rng: Random
    ) -> FilterSortParams | None:
        if kind is MutationKind.NUMBERS:
            return FilterSortParams(
                products=_products(rng, len(params.products)),
                threshold=rng.randint(5, 40),
                descending=params.descending,
            )
        if kind is MutationKind.INPUT_LENGTH:
            return FilterSortParams(
                products=_products(rng, len(params.products) + 2),
                threshold=params.threshold,
                descending=params.descending,
            )
        if kind is MutationKind.EDGE_CASE:
            top = max(p.price for p in params.products)
            return FilterSortParams(
                products=params.products, threshold=top + 1, descending=params.descending
            )
        if kind is MutationKind.FORMAT:
            return FilterSortParams(
                products=params.products,
                threshold=params.threshold,
                descending=not params.descending,
            )
        return None


class AggregateParams(TemplateParams):
    products: tuple[Product, ...] = Field(min_length=1)
    min_qty: int = Field(ge=0)


class AggregateTemplate(TaskTemplate[AggregateParams]):
    name = "aggregate"
    domain = TaskDomain.STRUCTURED_OUTPUT
    environment = "structured_output"
    verifier = "json_schema"
    params_model = AggregateParams
    mutations = frozenset({MutationKind.NUMBERS, MutationKind.INPUT_LENGTH, MutationKind.EDGE_CASE})
    _SCHEMA: ClassVar[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {"count": {"type": "integer"}, "total_value": {"type": "integer"}},
        "required": ["count", "total_value"],
        "additionalProperties": False,
    }

    @override
    def sample(self, rng: Random, difficulty: int) -> AggregateParams:
        return AggregateParams(products=_products(rng, 3 + difficulty), min_qty=rng.randint(0, 12))

    @override
    def build(self, params: AggregateParams) -> BuiltTask:
        selected = [p for p in params.products if p.qty > params.min_qty]
        expected: dict[str, JsonValue] = {
            "count": len(selected),
            "total_value": sum(p.price * p.qty for p in selected),
        }
        prompt = (
            f"Here is a JSON list of products:\n{_products_json(params.products)}\n\n"
            'Return a JSON object with keys "count" (the number of products with qty greater than '
            f'{params.min_qty}) and "total_value" (the sum of price * qty over those products).'
        )
        return BuiltTask(
            prompt=prompt,
            payload=StructuredOutputPayload(json_schema=dict(self._SCHEMA), expected=expected),
            signals={"operations": 3.0, "input_size": float(len(params.products))},
        )

    @override
    def mutate(
        self, params: AggregateParams, kind: MutationKind, rng: Random
    ) -> AggregateParams | None:
        if kind is MutationKind.NUMBERS:
            return AggregateParams(
                products=_products(rng, len(params.products)), min_qty=rng.randint(0, 12)
            )
        if kind is MutationKind.INPUT_LENGTH:
            return AggregateParams(
                products=_products(rng, len(params.products) + 2), min_qty=params.min_qty
            )
        if kind is MutationKind.EDGE_CASE:
            return AggregateParams(
                products=params.products, min_qty=max(p.qty for p in params.products)
            )
        return None


ShapeKind = Literal["sum2", "diff", "order", "prod", "neq", "sum3"]


class ConstraintShape(TemplateParams):
    kind: ShapeKind
    variables: tuple[int, ...] = Field(min_length=2, max_length=3)


class LogicParams(TemplateParams):
    names: tuple[str, ...] = Field(min_length=2, max_length=4)
    solution: tuple[int, ...]
    shapes: tuple[ConstraintShape, ...] = Field(min_length=1)
    upper: int = Field(default=9, ge=1, le=9)


def _render_constraint(
    shape: ConstraintShape, names: tuple[str, ...], solution: tuple[int, ...]
) -> str:
    v = [names[i] for i in shape.variables]
    s = [solution[i] for i in shape.variables]
    if shape.kind == "sum2":
        return f"{v[0]} + {v[1]} == {s[0] + s[1]}"
    if shape.kind == "diff":
        return f"{v[0]} - {v[1]} == {s[0] - s[1]}"
    if shape.kind == "prod":
        return f"{v[0]} * {v[1]} == {s[0] * s[1]}"
    if shape.kind == "neq":
        return f"{v[0]} != {v[1]}" if s[0] != s[1] else f"{v[0]} == {v[1]}"
    if shape.kind == "sum3":
        return f"{v[0]} + {v[1]} + {v[2]} == {s[0] + s[1] + s[2]}"
    relation = ">" if s[0] > s[1] else "<" if s[0] < s[1] else "=="
    return f"{v[0]} {relation} {v[1]}"


def _random_shape(rng: Random, n: int) -> ConstraintShape:
    kinds: list[ShapeKind] = ["sum2", "diff", "order", "prod", "neq"]
    if n >= 3:
        kinds.append("sum3")
    kind = rng.choice(kinds)
    arity = 3 if kind == "sum3" else 2
    return ConstraintShape(kind=kind, variables=tuple(rng.sample(range(n), arity)))


class LogicConstraintsTemplate(TaskTemplate[LogicParams]):
    """Constraint satisfaction with a hidden reference solution; any valid assignment passes."""

    name = "logic_constraints"
    domain = TaskDomain.STRUCTURED_OUTPUT
    environment = "structured_output"
    verifier = "json_schema"
    params_model = LogicParams
    mutations = frozenset(
        {MutationKind.CONSTRAINTS, MutationKind.NUMBERS, MutationKind.VARIABLE_NAMES}
    )

    @override
    def sample(self, rng: Random, difficulty: int) -> LogicParams:
        n = 2 + (difficulty >= 3) + (difficulty >= 5)
        solution = tuple(rng.randint(0, 9) for _ in range(n))
        shapes = [_random_shape(rng, n) for _ in range(n + difficulty // 2)]
        covered = {i for shape in shapes for i in shape.variables}
        for missing in sorted(set(range(n)) - covered):
            other = (missing + 1) % n
            shapes.append(ConstraintShape(kind="sum2", variables=(missing, other)))
        return LogicParams(names=VARIABLE_SETS[0][:n], solution=solution, shapes=tuple(shapes))

    @staticmethod
    def count_solutions(params: LogicParams, constraints: list[str]) -> int:
        count = 0
        for values in itertools.product(range(params.upper + 1), repeat=len(params.names)):
            variables = {
                name: Fraction(value) for name, value in zip(params.names, values, strict=True)
            }
            if all(evaluate_condition(c, variables) for c in constraints):
                count += 1
        return count

    @override
    def build(self, params: LogicParams) -> BuiltTask:
        constraints = [_render_constraint(s, params.names, params.solution) for s in params.shapes]
        names = ", ".join(params.names)
        prompt = (
            f"Find non-negative integers {names} (each at most {params.upper}) that satisfy all of "
            "these constraints:\n"
            + "\n".join(f"- {c}" for c in constraints)
            + "\n\nRespond with a "
            "JSON object that maps each variable name to its value."
        )
        schema: dict[str, JsonValue] = {
            "type": "object",
            "properties": {
                name: {"type": "integer", "minimum": 0, "maximum": params.upper}
                for name in params.names
            },
            "required": list(params.names),
            "additionalProperties": False,
        }
        reference: dict[str, JsonValue] = dict(zip(params.names, params.solution, strict=True))
        return BuiltTask(
            prompt=prompt,
            payload=StructuredOutputPayload(
                json_schema=schema,
                expected=None,
                constraints=tuple(constraints),
                reference_answer=reference,
            ),
            signals={
                "constraints": float(len(constraints)),
                "input_size": float(len(params.names)),
                "solutions": float(self.count_solutions(params, constraints)),
            },
        )

    @override
    def mutate(self, params: LogicParams, kind: MutationKind, rng: Random) -> LogicParams | None:
        n = len(params.names)
        if kind is MutationKind.CONSTRAINTS:
            return LogicParams(
                names=params.names,
                solution=params.solution,
                shapes=(*params.shapes, _random_shape(rng, n)),
                upper=params.upper,
            )
        if kind is MutationKind.NUMBERS:
            solution = tuple(rng.randint(0, params.upper) for _ in range(n))
            return LogicParams(
                names=params.names, solution=solution, shapes=params.shapes, upper=params.upper
            )
        if kind is MutationKind.VARIABLE_NAMES:
            options = [s[:n] for s in VARIABLE_SETS if s[:n] != params.names]
            return LogicParams(
                names=rng.choice(options),
                solution=params.solution,
                shapes=params.shapes,
                upper=params.upper,
            )
        return None


class PersonParams(TemplateParams):
    name: str
    age: int = Field(ge=0, le=120)
    city: str
    phrasing: int = Field(ge=0, le=2)


_PHRASINGS = (
    "{name} is {age} years old and lives in {city}.",
    "Living in {city}, {name} recently turned {age}.",
    "At {age}, {name} moved to {city}.",
)


class ExtractionTemplate(TaskTemplate[PersonParams]):
    name = "extraction"
    domain = TaskDomain.STRUCTURED_OUTPUT
    environment = "structured_output"
    verifier = "json_schema"
    params_model = PersonParams
    mutations = frozenset({MutationKind.VARIABLE_NAMES, MutationKind.NUMBERS, MutationKind.FORMAT})
    _SCHEMA: ClassVar[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "age", "city"],
        "additionalProperties": False,
    }

    @override
    def sample(self, rng: Random, difficulty: int) -> PersonParams:
        return PersonParams(
            name=rng.choice(PEOPLE),
            age=rng.randint(18, 90),
            city=rng.choice(CITIES),
            phrasing=rng.randrange(len(_PHRASINGS)),
        )

    @override
    def build(self, params: PersonParams) -> BuiltTask:
        sentence = _PHRASINGS[params.phrasing].format(
            name=params.name, age=params.age, city=params.city
        )
        prompt = (
            "Extract the person's name, age and city from the sentence below as a JSON object with "
            'keys "name" (string), "age" (integer) and "city" (string).\n\n'
            f"Sentence: {sentence}"
        )
        expected: dict[str, JsonValue] = {
            "name": params.name,
            "age": params.age,
            "city": params.city,
        }
        return BuiltTask(
            prompt=prompt,
            payload=StructuredOutputPayload(json_schema=dict(self._SCHEMA), expected=expected),
            signals={"operations": 1.0, "input_size": float(len(sentence.split()))},
        )

    @override
    def mutate(self, params: PersonParams, kind: MutationKind, rng: Random) -> PersonParams | None:
        data = params.model_dump()
        if kind is MutationKind.VARIABLE_NAMES:
            data.update(name=rng.choice(PEOPLE), city=rng.choice(CITIES))
        elif kind is MutationKind.NUMBERS:
            data.update(age=rng.randint(18, 90))
        elif kind is MutationKind.FORMAT:
            data.update(phrasing=(params.phrasing + 1) % len(_PHRASINGS))
        else:
            return None
        return PersonParams.model_validate(data)
