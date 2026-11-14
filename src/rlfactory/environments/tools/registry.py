"""Deterministic tools (calculator, key-value lookup, JSON transform) and their registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol

from pydantic import JsonValue

from rlfactory.domain.numeric import ExpressionError, canonical_number, evaluate_arithmetic
from rlfactory.errors import RegistryError


class ToolUsageError(ValueError):
    """The model called a tool incorrectly (bad arguments, unknown key, ...)."""


@dataclass(frozen=True)
class ToolState:
    """Per-episode tool state, built from the task payload."""

    kv_store: Mapping[str, JsonValue] = field(default_factory=dict)
    document: JsonValue = None


class Tool(Protocol):
    name: str
    description: str
    arguments: Mapping[str, str]
    """Argument name -> short type/description, rendered into the prompt."""

    def __call__(self, arguments: Mapping[str, JsonValue], state: ToolState) -> JsonValue: ...


def _require_str(arguments: Mapping[str, JsonValue], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ToolUsageError(f'argument "{key}" must be a string')
    return value


class CalculatorTool:
    name = "calculator"
    description = "Evaluate an arithmetic expression exactly (+ - * / // % ** abs min max)."
    arguments: Mapping[str, str] = {"expression": "string"}

    def __call__(
        self,
        arguments: Mapping[str, JsonValue],
        state: ToolState,  # noqa: ARG002 - Tool protocol signature
    ) -> JsonValue:
        expression = _require_str(arguments, "expression")
        try:
            return canonical_number(evaluate_arithmetic(expression))
        except ExpressionError as exc:
            raise ToolUsageError(str(exc)) from exc


class KeyValueLookupTool:
    name = "kv_lookup"
    description = "Look up a value by exact key in the task's key-value store."
    arguments: Mapping[str, str] = {"key": "string"}

    def __call__(self, arguments: Mapping[str, JsonValue], state: ToolState) -> JsonValue:
        key = _require_str(arguments, "key")
        if key not in state.kv_store:
            raise ToolUsageError(f"unknown key {key!r}")
        return state.kv_store[key]


def _resolve_path(document: JsonValue, path: str) -> JsonValue:
    current = document
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise ToolUsageError(f"path segment {part!r} not found")
            current = current[part]
        elif isinstance(current, list):
            if not part.lstrip("-").isdigit():
                raise ToolUsageError(f"list index expected, got {part!r}")
            index = int(part)
            if not -len(current) <= index < len(current):
                raise ToolUsageError(f"index {index} out of range")
            current = current[index]
        else:
            raise ToolUsageError(f"cannot descend into {type(current).__name__}")
    return current


def _matches(item: JsonValue, where: Mapping[str, JsonValue]) -> bool:
    return isinstance(item, dict) and all(item.get(k) == v for k, v in where.items())


class JsonTransformTool:
    name = "json_transform"
    description = (
        "Query the task's JSON document. op is one of get, keys, count, sum, sort, filter; "
        'path is dot-separated (e.g. "orders" or "orders.0.amount"); optional field, '
        "where (object of field equalities) and descending (bool)."
    )
    arguments: Mapping[str, str] = {
        "op": "string",
        "path": "string",
        "field": "string (sum/sort)",
        "where": "object (optional)",
        "descending": "boolean (optional)",
    }
    _OPS = frozenset({"get", "keys", "count", "sum", "sort", "filter"})

    def __call__(self, arguments: Mapping[str, JsonValue], state: ToolState) -> JsonValue:
        op = _require_str(arguments, "op")
        if op not in self._OPS:
            raise ToolUsageError(f"unknown op {op!r}")
        path = arguments.get("path", "")
        if not isinstance(path, str):
            raise ToolUsageError('"path" must be a string')
        target = _resolve_path(state.document, path)
        if op == "get":
            return target
        if op == "keys":
            if not isinstance(target, dict):
                raise ToolUsageError("keys requires an object")
            keys: list[JsonValue] = [str(key) for key in sorted(target)]
            return keys
        if not isinstance(target, list):
            raise ToolUsageError(f"{op} requires a list")
        where = arguments.get("where", {})
        if not isinstance(where, dict):
            raise ToolUsageError('"where" must be an object')
        items = [item for item in target if _matches(item, where)] if where else list(target)
        if op == "count":
            return len(items)
        if op == "filter":
            return items
        field_name = _require_str(arguments, "field")
        if op == "sum":
            total = Fraction(0)
            for item in items:
                value = item.get(field_name) if isinstance(item, dict) else None
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ToolUsageError(f"field {field_name!r} is not numeric in every item")
                total += Fraction(repr(value))
            return canonical_number(total)
        descending = arguments.get("descending", False) is True
        try:
            return sorted(
                items,
                key=lambda item: item[field_name] if isinstance(item, dict) else 0,  # type: ignore[arg-type,return-value]
                reverse=descending,
            )
        except (KeyError, TypeError) as exc:
            raise ToolUsageError(f"cannot sort by {field_name!r}") from exc


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise RegistryError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise RegistryError(f"unknown tool {name!r}") from exc

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self, names: tuple[str, ...]) -> str:
        lines = []
        for name in names:
            tool = self.get(name)
            args = ", ".join(f'"{k}": {v}' for k, v in tool.arguments.items())
            lines.append(f"- {tool.name}: {tool.description} Arguments: {{{args}}}")
        return "\n".join(lines)


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(KeyValueLookupTool())
    registry.register(JsonTransformTool())
    return registry
