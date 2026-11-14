"""Tool-use templates: facts live behind tools, so answers require multi-step tool calls."""

from __future__ import annotations

from fractions import Fraction
from random import Random
from typing import Literal

from pydantic import Field, JsonValue, model_validator
from typing_extensions import override

from rlfactory.domain.numeric import canonical_number
from rlfactory.domain.task import TaskDomain, ToolCall, ToolUsePayload
from rlfactory.tasks.generators.base import BuiltTask, MutationKind, TaskTemplate, TemplateParams
from rlfactory.tasks.generators.vocabulary import CURRENCIES, PEOPLE, PRODUCTS


class StockItem(TemplateParams):
    name: str
    price: int = Field(ge=0)
    stock: int = Field(ge=0)


class InventoryParams(TemplateParams):
    items: tuple[StockItem, ...] = Field(min_length=1)
    query: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _query_known(self) -> InventoryParams:
        known = {item.name for item in self.items}
        if not set(self.query) <= known or len(set(self.query)) != len(self.query):
            raise ValueError("query must name distinct known items")
        return self


def _stock(rng: Random, names: list[str]) -> tuple[StockItem, ...]:
    return tuple(
        StockItem(name=n, price=rng.randint(1, 30), stock=rng.randint(0, 50)) for n in names
    )


class InventoryValueTemplate(TaskTemplate[InventoryParams]):
    name = "inventory_value"
    domain = TaskDomain.TOOL_USE
    environment = "tool_use"
    verifier = "tool_outcome"
    params_model = InventoryParams
    mutations = frozenset(
        {
            MutationKind.NUMBERS,
            MutationKind.INPUT_LENGTH,
            MutationKind.VARIABLE_NAMES,
            MutationKind.EDGE_CASE,
        }
    )

    @override
    def sample(self, rng: Random, difficulty: int) -> InventoryParams:
        names = rng.sample(PRODUCTS, k=4 + difficulty // 2)
        query_size = min(len(names), 1 + (difficulty + 1) // 2)
        return InventoryParams(
            items=_stock(rng, names), query=tuple(rng.sample(names, k=query_size))
        )

    @override
    def build(self, params: InventoryParams) -> BuiltTask:
        by_name = {item.name: item for item in params.items}
        kv: dict[str, JsonValue] = {}
        for item in params.items:
            kv[f"price:{item.name}"] = item.price
            kv[f"stock:{item.name}"] = item.stock
        plan: list[ToolCall] = []
        for name in params.query:
            plan.append(ToolCall(name="kv_lookup", arguments={"key": f"price:{name}"}))
            plan.append(ToolCall(name="kv_lookup", arguments={"key": f"stock:{name}"}))
        expression = " + ".join(f"{by_name[n].price}*{by_name[n].stock}" for n in params.query)
        plan.append(ToolCall(name="calculator", arguments={"expression": expression}))
        total = sum(by_name[n].price * by_name[n].stock for n in params.query)
        prompt = (
            "What is the total stock value (price times stock, summed over items) of: "
            f"{', '.join(params.query)}? Prices and stock levels are only available through the "
            'kv_lookup tool, with keys like "price:<item>" and "stock:<item>". Use the calculator '
            "for arithmetic."
        )
        return BuiltTask(
            prompt=prompt,
            payload=ToolUsePayload(
                tools=("calculator", "kv_lookup"),
                kv_store=kv,
                expected_answer=str(total),
                required_tools=("calculator", "kv_lookup"),
                max_steps=len(plan) + 3,
                reference_plan=tuple(plan),
            ),
            signals={"tool_steps": float(len(plan)), "input_size": float(len(params.items))},
        )

    @override
    def mutate(
        self, params: InventoryParams, kind: MutationKind, rng: Random
    ) -> InventoryParams | None:
        names = [item.name for item in params.items]
        if kind is MutationKind.NUMBERS:
            return InventoryParams(items=_stock(rng, names), query=params.query)
        if kind is MutationKind.INPUT_LENGTH:
            remaining = [n for n in names if n not in params.query]
            if not remaining:
                return None
            return InventoryParams(items=params.items, query=(*params.query, rng.choice(remaining)))
        if kind is MutationKind.VARIABLE_NAMES:
            fresh = rng.sample(
                [p for p in PRODUCTS if p not in names], k=min(len(names), 20 - len(names))
            )
            if len(fresh) < len(names):
                return None
            mapping = dict(zip(names, fresh, strict=False))
            items = tuple(
                item.model_copy(update={"name": mapping[item.name]}) for item in params.items
            )
            return InventoryParams(items=items, query=tuple(mapping[q] for q in params.query))
        if kind is MutationKind.EDGE_CASE:
            target = params.query[0]
            items = tuple(
                item.model_copy(update={"stock": 0}) if item.name == target else item
                for item in params.items
            )
            return InventoryParams(items=items, query=params.query)
        return None


Status = Literal["shipped", "pending", "cancelled"]
_STATUSES: tuple[Status, ...] = ("shipped", "pending", "cancelled")


class Order(TemplateParams):
    id: int
    customer: str
    amount: int = Field(ge=0)
    status: Status


class OrderParams(TemplateParams):
    orders: tuple[Order, ...] = Field(min_length=1)
    status: Status


def _orders(rng: Random, count: int) -> tuple[Order, ...]:
    return tuple(
        Order(
            id=i + 1,
            customer=rng.choice(PEOPLE),
            amount=rng.randint(5, 200),
            status=rng.choice(_STATUSES),
        )
        for i in range(count)
    )


class OrderTotalTemplate(TaskTemplate[OrderParams]):
    name = "order_total"
    domain = TaskDomain.TOOL_USE
    environment = "tool_use"
    verifier = "tool_outcome"
    params_model = OrderParams
    mutations = frozenset({MutationKind.NUMBERS, MutationKind.INPUT_LENGTH, MutationKind.EDGE_CASE})

    @override
    def sample(self, rng: Random, difficulty: int) -> OrderParams:
        return OrderParams(orders=_orders(rng, 4 + 2 * difficulty), status=rng.choice(_STATUSES))

    @override
    def build(self, params: OrderParams) -> BuiltTask:
        document: dict[str, JsonValue] = {"orders": [order.model_dump() for order in params.orders]}
        total = sum(order.amount for order in params.orders if order.status == params.status)
        plan = (
            ToolCall(
                name="json_transform",
                arguments={
                    "op": "sum",
                    "path": "orders",
                    "field": "amount",
                    "where": {"status": params.status},
                },
            ),
        )
        prompt = (
            "An order document is available only through the json_transform tool. What is the "
            f'total amount of all orders whose status is "{params.status}"?'
        )
        return BuiltTask(
            prompt=prompt,
            payload=ToolUsePayload(
                tools=("json_transform", "calculator"),
                document=document,
                expected_answer=canonical_number(Fraction(total)),
                required_tools=("json_transform",),
                max_steps=4,
                reference_plan=plan,
            ),
            signals={"tool_steps": 1.0, "input_size": float(len(params.orders))},
        )

    @override
    def mutate(self, params: OrderParams, kind: MutationKind, rng: Random) -> OrderParams | None:
        if kind is MutationKind.NUMBERS:
            orders = tuple(
                o.model_copy(update={"amount": rng.randint(5, 200)}) for o in params.orders
            )
            return OrderParams(orders=orders, status=params.status)
        if kind is MutationKind.INPUT_LENGTH:
            extra = _orders(rng, 3)
            renumbered = tuple(
                o.model_copy(update={"id": len(params.orders) + i + 1}) for i, o in enumerate(extra)
            )
            return OrderParams(orders=(*params.orders, *renumbered), status=params.status)
        if kind is MutationKind.EDGE_CASE:
            other: Status = next(s for s in _STATUSES if s != params.status)
            orders = tuple(
                o.model_copy(update={"status": other}) if o.status == params.status else o
                for o in params.orders
            )
            return OrderParams(orders=orders, status=params.status)
        return None


_RATES = ("0.5", "0.75", "0.8", "0.85", "0.9", "1.1", "1.25", "1.5", "2", "2.5", "4")


class ConversionParams(TemplateParams):
    amount: int = Field(ge=0)
    source: str
    target: str
    rate: str
    distractors: tuple[tuple[str, str, str], ...] = ()


class CurrencyConversionTemplate(TaskTemplate[ConversionParams]):
    name = "currency_conversion"
    domain = TaskDomain.TOOL_USE
    environment = "tool_use"
    verifier = "tool_outcome"
    params_model = ConversionParams
    mutations = frozenset(
        {MutationKind.NUMBERS, MutationKind.VARIABLE_NAMES, MutationKind.EDGE_CASE}
    )

    @staticmethod
    def _distractors(rng: Random, source: str, target: str) -> tuple[tuple[str, str, str], ...]:
        pairs = [
            (a, b) for a in CURRENCIES for b in CURRENCIES if a != b and (a, b) != (source, target)
        ]
        return tuple((a, b, rng.choice(_RATES)) for a, b in rng.sample(pairs, k=2))

    @override
    def sample(self, rng: Random, difficulty: int) -> ConversionParams:
        source, target = rng.sample(CURRENCIES, k=2)
        return ConversionParams(
            amount=rng.randint(10, 200 * difficulty),
            source=source,
            target=target,
            rate=rng.choice(_RATES),
            distractors=self._distractors(rng, source, target),
        )

    @override
    def build(self, params: ConversionParams) -> BuiltTask:
        key = f"rate:{params.source}_to_{params.target}"
        kv: dict[str, JsonValue] = {key: params.rate}
        for a, b, rate in params.distractors:
            kv[f"rate:{a}_to_{b}"] = rate
        value = params.amount * Fraction(params.rate)
        plan = (
            ToolCall(name="kv_lookup", arguments={"key": key}),
            ToolCall(
                name="calculator", arguments={"expression": f"{params.amount} * {params.rate}"}
            ),
        )
        prompt = (
            f"Convert {params.amount} {params.source.upper()} to {params.target.upper()}. Exchange "
            'rates are only available through kv_lookup with keys like "rate:<from>_to_<to>" using '
            "lowercase currency codes. Give the exact converted amount."
        )
        return BuiltTask(
            prompt=prompt,
            payload=ToolUsePayload(
                tools=("calculator", "kv_lookup"),
                kv_store=kv,
                expected_answer=canonical_number(value),
                required_tools=("calculator", "kv_lookup"),
                max_steps=5,
                reference_plan=plan,
            ),
            signals={"tool_steps": 2.0, "max_operand": float(params.amount)},
        )

    @override
    def mutate(
        self, params: ConversionParams, kind: MutationKind, rng: Random
    ) -> ConversionParams | None:
        data = params.model_dump()
        if kind is MutationKind.NUMBERS:
            data.update(amount=rng.randint(10, 1000), rate=rng.choice(_RATES))
        elif kind is MutationKind.VARIABLE_NAMES:
            source, target = rng.sample(CURRENCIES, k=2)
            data.update(
                source=source, target=target, distractors=self._distractors(rng, source, target)
            )
        elif kind is MutationKind.EDGE_CASE:
            data.update(rate="1")
        else:
            return None
        return ConversionParams.model_validate(data)
