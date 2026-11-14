from __future__ import annotations

from fractions import Fraction

import pytest

from rlfactory.domain.numeric import (
    ExpressionError,
    canonical_number,
    evaluate_arithmetic,
    evaluate_condition,
    last_number,
    parse_number,
)
from rlfactory.domain.task import ToolCall
from rlfactory.environments.parsing import (
    extract_code_block,
    extract_json_answer,
    extract_tag,
    parse_tool_call,
    render_tool_call,
)
from rlfactory.environments.tools.registry import (
    CalculatorTool,
    JsonTransformTool,
    KeyValueLookupTool,
    ToolState,
    ToolUsageError,
    default_tool_registry,
)
from rlfactory.errors import RegistryError


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("17", Fraction(17)),
        (" -3/4 ", Fraction(-3, 4)),
        ("1,234.5", Fraction(24690, 20)),
        ("1e-3", Fraction(1, 1000)),
        ("$12.", Fraction(12)),
        ("\\boxed{42}", Fraction(42)),
        ("\\frac{3}{4}", Fraction(3, 4)),
        ("-\\frac{1}{2}", Fraction(-1, 2)),
        ("13.50", Fraction(27, 2)),
    ],
)
def test_parse_number_known_values(text: str, expected: Fraction) -> None:
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "1/0", "nan", "inf", "1,2", "3 apples"])
def test_parse_number_rejects_non_numbers(text: str) -> None:
    assert parse_number(text) is None


def test_last_number_is_lenient() -> None:
    assert last_number("first 3 then 7/2 finally 12") == 12
    assert last_number("no digits") is None


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (Fraction(12), "12"),
        (Fraction(27, 2), "13.5"),
        (Fraction(1, 3), "1/3"),
        (Fraction(-1, 8), "-0.125"),
        (Fraction(0), "0"),
        (Fraction(-7, 20), "-0.35"),
    ],
)
def test_canonical_number(value: Fraction, text: str) -> None:
    assert canonical_number(value) == text
    assert parse_number(text) == value


def test_evaluate_arithmetic_exact() -> None:
    assert evaluate_arithmetic("1/3 + 1/6") == Fraction(1, 2)
    assert evaluate_arithmetic("2 ** 10 - abs(-4) * max(1, 2)") == 1016
    assert evaluate_arithmetic("0.1 + 0.2") == Fraction(3, 10)
    assert evaluate_arithmetic("x * 3", {"x": Fraction(4)}) == 12


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "().__class__",
        "[1, 2][0]",
        "1 / 0",
        "2 ** 100",
        "lambda: 1",
        "open('x')",
        "True + 1",
        "'a' * 3",
        "1 if 2 else 3",
        "x",
    ],
)
def test_evaluate_arithmetic_rejects_unsafe_or_invalid(expression: str) -> None:
    with pytest.raises(ExpressionError):
        evaluate_arithmetic(expression)


def test_evaluate_condition() -> None:
    values = {"x": Fraction(3), "y": Fraction(7)}
    assert evaluate_condition("x + y == 10 and y > x", values)
    assert not evaluate_condition("x > y or x == 4", values)
    assert evaluate_condition("not x == y", values)
    assert evaluate_condition("1 < x < y", values)
    with pytest.raises(ExpressionError):
        evaluate_condition("x + y", values)


def test_value_bounds_enforced() -> None:
    with pytest.raises(ExpressionError, match="bounds"):
        evaluate_arithmetic("10 ** 12 * 10 ** 12 * 10 ** 12")


def test_extract_tag_last_occurrence_wins() -> None:
    assert extract_tag("<answer>1</answer> then <answer> 2 </answer>") == "2"
    assert extract_tag("no tags") is None
    assert extract_tag("<answer>\nmulti\nline\n</answer>") == "multi\nline"
    with pytest.raises(ValueError, match="invalid tag"):
        extract_tag("x", "bad tag")


def test_extract_code_block_prefers_language() -> None:
    text = "```\nuntagged()\n```\n```python\ndef f():\n    return 1\n```"
    assert extract_code_block(text) == "def f():\n    return 1"
    assert extract_code_block("```\nonly_untagged()\n```") == "only_untagged()"
    assert extract_code_block("no code") is None
    assert extract_code_block("```python\n   \n```") is None


def test_extract_json_answer() -> None:
    assert extract_json_answer('<answer>```json\n{"a": 1}\n```</answer>') == '{"a": 1}'
    assert extract_json_answer("```json\n[1, 2]\n```") == "[1, 2]"
    assert extract_json_answer("nothing") is None


def test_parse_tool_call_variants() -> None:
    call = ToolCall(name="calculator", arguments={"expression": "1+1"})
    assert parse_tool_call(render_tool_call(call)) == (call, None)
    assert parse_tool_call("plain text") == (None, None)
    assert parse_tool_call("<tool_call>{bad</tool_call>")[1] == "tool call is not valid JSON"
    assert parse_tool_call('<tool_call>{"arguments": {}}</tool_call>')[1] is not None
    assert parse_tool_call('<tool_call>{"name": "x", "arguments": [1]}</tool_call>')[1] is not None


def test_calculator_tool() -> None:
    tool = CalculatorTool()
    assert tool({"expression": "3*12 + 5*7"}, ToolState()) == "71"
    assert tool({"expression": "120 * 0.85"}, ToolState()) == "102"
    with pytest.raises(ToolUsageError):
        tool({"expression": "__import__('os')"}, ToolState())
    with pytest.raises(ToolUsageError, match="string"):
        tool({"expression": 5}, ToolState())


def test_kv_lookup_tool() -> None:
    tool = KeyValueLookupTool()
    state = ToolState(kv_store={"price:apple": 3})
    assert tool({"key": "price:apple"}, state) == 3
    with pytest.raises(ToolUsageError, match="unknown key"):
        tool({"key": "price:pear"}, state)


def test_json_transform_tool() -> None:
    tool = JsonTransformTool()
    state = ToolState(
        document={
            "orders": [
                {"id": 1, "amount": 30, "status": "shipped"},
                {"id": 2, "amount": 12, "status": "pending"},
                {"id": 3, "amount": 8, "status": "shipped"},
            ]
        }
    )
    assert (
        tool(
            {"op": "sum", "path": "orders", "field": "amount", "where": {"status": "shipped"}},
            state,
        )
        == "38"
    )
    assert tool({"op": "count", "path": "orders"}, state) == 3
    assert tool({"op": "get", "path": "orders.1.amount"}, state) == 12
    assert tool({"op": "keys", "path": "orders.0"}, state) == ["amount", "id", "status"]
    ordered = tool({"op": "sort", "path": "orders", "field": "amount", "descending": True}, state)
    assert [o["id"] for o in ordered] == [1, 2, 3]  # type: ignore[index,union-attr]
    for bad in (
        {"op": "drop", "path": "orders"},
        {"op": "get", "path": "orders.9"},
        {"op": "keys", "path": "orders"},
        {"op": "sum", "path": "orders", "field": "status"},
    ):
        with pytest.raises(ToolUsageError):
            tool(bad, state)


def test_tool_registry() -> None:
    registry = default_tool_registry()
    assert registry.names() == ["calculator", "json_transform", "kv_lookup"]
    assert "calculator" in registry
    assert "calculator:" in registry.describe(("calculator",))
    with pytest.raises(RegistryError):
        registry.get("shell")
    with pytest.raises(RegistryError, match="already"):
        registry.register(CalculatorTool())
