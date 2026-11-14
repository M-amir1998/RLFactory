"""Arithmetic templates: integer chains, fraction sums, word problems and percentages."""

from __future__ import annotations

from fractions import Fraction
from random import Random
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import override

from rlfactory.domain.numeric import canonical_number, evaluate_arithmetic
from rlfactory.domain.task import ArithmeticPayload, TaskDomain
from rlfactory.tasks.generators.base import BuiltTask, MutationKind, TaskTemplate, TemplateParams
from rlfactory.tasks.generators.vocabulary import PRODUCTS

Operator = Literal["+", "-", "*"]
Sign = Literal["+", "-"]
_SIGNS: tuple[Sign, ...] = ("+", "-")


def _operand(value: int) -> str:
    return f"({value})" if value < 0 else str(value)


class ChainParams(TemplateParams):
    operands: tuple[int, ...] = Field(min_length=2)
    operators: tuple[Operator, ...]

    @model_validator(mode="after")
    def _arity(self) -> ChainParams:
        if len(self.operators) != len(self.operands) - 1:
            raise ValueError("need exactly one operator between consecutive operands")
        return self


class IntegerChainTemplate(TaskTemplate[ChainParams]):
    name = "integer_chain"
    domain = TaskDomain.ARITHMETIC
    environment = "arithmetic"
    verifier = "numeric"
    params_model = ChainParams
    mutations = frozenset({MutationKind.NUMBERS, MutationKind.INPUT_LENGTH, MutationKind.EDGE_CASE})

    @override
    def sample(self, rng: Random, difficulty: int) -> ChainParams:
        count = min(6, 1 + difficulty)
        high = 10 ** min(3, 1 + difficulty // 2)
        operators: list[Operator] = []
        for _ in range(count):
            operator: Operator = rng.choice(("+", "-", "*"))
            if operator == "*" and operators.count("*") >= 2:
                operator = rng.choice(("+", "-"))
            operators.append(operator)
        operands = tuple(rng.randint(1, high) for _ in range(count + 1))
        return ChainParams(operands=operands, operators=tuple(operators))

    @override
    def build(self, params: ChainParams) -> BuiltTask:
        parts = [_operand(params.operands[0])]
        for operator, operand in zip(params.operators, params.operands[1:], strict=True):
            parts += [operator, _operand(operand)]
        expression = " ".join(parts)
        answer = canonical_number(evaluate_arithmetic(expression))
        return BuiltTask(
            prompt=f"Compute the value of {expression}.",
            payload=ArithmeticPayload(expression=expression, answer=answer),
            signals={
                "operations": float(len(params.operators)),
                "max_operand": float(max(abs(o) for o in params.operands)),
            },
        )

    @override
    def mutate(self, params: ChainParams, kind: MutationKind, rng: Random) -> ChainParams | None:
        high = max(10, max(abs(o) for o in params.operands))
        if kind is MutationKind.NUMBERS:
            operands = tuple(rng.randint(1, high) for _ in params.operands)
            return ChainParams(operands=operands, operators=params.operators)
        if kind is MutationKind.INPUT_LENGTH:
            extra: Operator = rng.choice(("+", "-"))
            return ChainParams(
                operands=(*params.operands, rng.randint(1, high)),
                operators=(*params.operators, extra),
            )
        if kind is MutationKind.EDGE_CASE:
            values = list(params.operands)
            values[rng.randrange(len(values))] = rng.choice((0, -rng.randint(1, high)))
            return ChainParams(operands=tuple(values), operators=params.operators)
        return None


class FractionParams(TemplateParams):
    numerators: tuple[int, ...] = Field(min_length=2)
    denominators: tuple[int, ...] = Field(min_length=2)
    signs: tuple[Literal["+", "-"], ...]

    @model_validator(mode="after")
    def _shape(self) -> FractionParams:
        if (
            len(self.numerators) != len(self.denominators)
            or len(self.signs) != len(self.numerators) - 1
        ):
            raise ValueError("inconsistent fraction term counts")
        if any(d <= 0 for d in self.denominators):
            raise ValueError("denominators must be positive")
        return self


class FractionSumTemplate(TaskTemplate[FractionParams]):
    name = "fraction_sum"
    domain = TaskDomain.ARITHMETIC
    environment = "arithmetic"
    verifier = "numeric"
    params_model = FractionParams
    mutations = frozenset({MutationKind.NUMBERS, MutationKind.INPUT_LENGTH, MutationKind.EDGE_CASE})

    @override
    def sample(self, rng: Random, difficulty: int) -> FractionParams:
        terms = 2 if difficulty < 4 else 3
        denominators = tuple(rng.randint(2, 4 + 2 * difficulty) for _ in range(terms))
        numerators = tuple(rng.randint(1, 2 * d) for d in denominators)
        signs: tuple[Sign, ...] = tuple(rng.choice(_SIGNS) for _ in range(terms - 1))
        return FractionParams(numerators=numerators, denominators=denominators, signs=signs)

    @override
    def build(self, params: FractionParams) -> BuiltTask:
        parts = [f"{params.numerators[0]}/{params.denominators[0]}"]
        for sign, n, d in zip(
            params.signs, params.numerators[1:], params.denominators[1:], strict=True
        ):
            parts += [sign, f"{n}/{d}"]
        expression = " ".join(parts)
        answer = canonical_number(evaluate_arithmetic(expression))
        return BuiltTask(
            prompt=(
                f"Compute {expression}. Give the exact result as an integer, a terminating decimal "
                "or a reduced fraction."
            ),
            payload=ArithmeticPayload(expression=expression, answer=answer),
            signals={
                "operations": float(2 * len(params.numerators) - 1),
                "max_operand": float(max(params.denominators)),
            },
        )

    @override
    def mutate(
        self, params: FractionParams, kind: MutationKind, rng: Random
    ) -> FractionParams | None:
        if kind is MutationKind.NUMBERS:
            numerators = tuple(rng.randint(1, 2 * d) for d in params.denominators)
            return FractionParams(
                numerators=numerators, denominators=params.denominators, signs=params.signs
            )
        if kind is MutationKind.INPUT_LENGTH:
            d = rng.randint(2, 12)
            return FractionParams(
                numerators=(*params.numerators, rng.randint(1, 2 * d)),
                denominators=(*params.denominators, d),
                signs=(*params.signs, rng.choice(("+", "-"))),
            )
        if kind is MutationKind.EDGE_CASE:
            # Two terms with equal denominators whose sum is a whole number.
            d = params.denominators[0]
            first = rng.randint(1, d - 1) if d > 1 else 1
            return FractionParams(numerators=(first, d - first), denominators=(d, d), signs=("+",))
        return None


class WarehouseParams(TemplateParams):
    crates: int = Field(ge=0)
    delivered: int = Field(ge=0)
    shipped: int = Field(ge=0)
    per_crate: int = Field(ge=1)
    item: str

    @model_validator(mode="after")
    def _feasible(self) -> WarehouseParams:
        if self.shipped > self.crates + self.delivered:
            raise ValueError("cannot ship more crates than available")
        return self


class WordProblemTemplate(TaskTemplate[WarehouseParams]):
    name = "warehouse_word_problem"
    domain = TaskDomain.ARITHMETIC
    environment = "arithmetic"
    verifier = "numeric"
    params_model = WarehouseParams
    mutations = frozenset(
        {MutationKind.NUMBERS, MutationKind.EDGE_CASE, MutationKind.VARIABLE_NAMES}
    )

    @override
    def sample(self, rng: Random, difficulty: int) -> WarehouseParams:
        crates = rng.randint(5, 20 * difficulty)
        delivered = rng.randint(1, 10 * difficulty)
        return WarehouseParams(
            crates=crates,
            delivered=delivered,
            shipped=rng.randint(0, crates + delivered),
            per_crate=rng.randint(2, 6 + 4 * difficulty),
            item=rng.choice(PRODUCTS),
        )

    @override
    def build(self, params: WarehouseParams) -> BuiltTask:
        remaining = (params.crates + params.delivered - params.shipped) * params.per_crate
        item = f"{params.item}s"
        prompt = (
            f"A warehouse holds {params.crates} crates of {item}. A truck delivers "
            f"{params.delivered} more crates, then {params.shipped} crates are shipped out. Each "
            f"crate contains {params.per_crate} {item}. How many {item} remain in the warehouse?"
        )
        expression = (
            f"({params.crates} + {params.delivered} - {params.shipped}) * {params.per_crate}"
        )
        return BuiltTask(
            prompt=prompt,
            payload=ArithmeticPayload(expression=expression, answer=str(remaining)),
            signals={
                "operations": 3.0,
                "max_operand": float(max(params.crates, params.delivered, params.per_crate)),
            },
        )

    @override
    def mutate(
        self, params: WarehouseParams, kind: MutationKind, rng: Random
    ) -> WarehouseParams | None:
        if kind is MutationKind.NUMBERS:
            crates = rng.randint(1, max(10, params.crates * 2))
            delivered = rng.randint(0, max(5, params.delivered * 2))
            return params.model_validate(
                {
                    **params.model_dump(),
                    "crates": crates,
                    "delivered": delivered,
                    "shipped": rng.randint(0, crates + delivered),
                }
            )
        if kind is MutationKind.EDGE_CASE:
            return params.model_validate(
                {**params.model_dump(), "shipped": params.crates + params.delivered}
            )
        if kind is MutationKind.VARIABLE_NAMES:
            options = [p for p in PRODUCTS if p != params.item]
            return params.model_validate({**params.model_dump(), "item": rng.choice(options)})
        return None


class PercentParams(TemplateParams):
    percent: int = Field(ge=0, le=500)
    base: int = Field(ge=0)


class PercentageTemplate(TaskTemplate[PercentParams]):
    name = "percentage"
    domain = TaskDomain.ARITHMETIC
    environment = "arithmetic"
    verifier = "numeric"
    params_model = PercentParams
    mutations = frozenset({MutationKind.NUMBERS, MutationKind.EDGE_CASE})
    _COMMON = (5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90, 125, 150)

    @override
    def sample(self, rng: Random, difficulty: int) -> PercentParams:
        percent = rng.choice(self._COMMON) if difficulty < 3 else rng.randint(1, 199)
        return PercentParams(percent=percent, base=rng.randint(10, 100 * difficulty))

    @override
    def build(self, params: PercentParams) -> BuiltTask:
        value = Fraction(params.percent * params.base, 100)
        return BuiltTask(
            prompt=f"What is {params.percent}% of {params.base}? Give the exact value.",
            payload=ArithmeticPayload(
                expression=f"{params.percent} / 100 * {params.base}", answer=canonical_number(value)
            ),
            signals={"operations": 2.0, "max_operand": float(params.base)},
        )

    @override
    def mutate(
        self, params: PercentParams, kind: MutationKind, rng: Random
    ) -> PercentParams | None:
        if kind is MutationKind.NUMBERS:
            return PercentParams(
                percent=rng.randint(1, 199), base=rng.randint(10, max(20, params.base * 2))
            )
        if kind is MutationKind.EDGE_CASE:
            return PercentParams(percent=rng.choice((0, 100)), base=params.base)
        return None
