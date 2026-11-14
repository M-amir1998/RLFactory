"""Exact, safe arithmetic: number parsing, canonical formatting and a restricted evaluator.

The evaluator interprets a tiny, explicit subset of Python expression syntax over
:class:`fractions.Fraction` values. It never calls :func:`eval`, rejects attribute access,
subscripts, lambdas and comprehensions, and bounds exponents, magnitudes and expression size.
It is shared by task generators, the calculator tool and constraint checking.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable, Mapping
from fractions import Fraction

MAX_ABS_VALUE = Fraction(10) ** 30
MAX_EXPONENT = 12
MAX_NODES = 256


class ExpressionError(ValueError):
    """Raised for unsupported syntax, invalid operations or out-of-bounds values."""


_THOUSANDS = re.compile(r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$")
_FRACTION = re.compile(r"^([-+]?\d+)\s*/\s*([-+]?\d+)$")
_LATEX_FRAC = re.compile(r"^([-+]?)\\[dt]?frac\{\s*([-+]?\d+)\s*\}\{\s*([-+]?\d+)\s*\}$")
_BOXED = re.compile(r"\\boxed\{(.*)\}")
_NUMBER_TOKEN = re.compile(
    r"[-+]?\d+\s*/\s*\d+|[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+"
)


def parse_number(text: str) -> Fraction | None:
    """Parse an integer, decimal, scientific, fraction (``-3/4``) or LaTeX ``\\frac`` literal.

    Surrounding whitespace, a leading ``$``, a trailing period and ``\\boxed{}`` are tolerated.
    Returns ``None`` when ``text`` is not exactly one number.
    """
    candidate = text.strip()
    boxed = _BOXED.search(candidate)
    if boxed:
        candidate = boxed.group(1).strip()
    candidate = candidate.removeprefix("$").removesuffix("$").strip().rstrip(".").strip()
    if not candidate:
        return None
    latex = _LATEX_FRAC.match(candidate)
    if latex:
        sign, num, den = latex.groups()
        if int(den) == 0:
            return None
        value = Fraction(int(num), int(den))
        return -value if sign == "-" else value
    fraction = _FRACTION.match(candidate)
    if fraction:
        num, den = int(fraction.group(1)), int(fraction.group(2))
        return None if den == 0 else Fraction(num, den)
    if _THOUSANDS.match(candidate):
        candidate = candidate.replace(",", "")
    try:
        return Fraction(candidate)
    except (ValueError, ZeroDivisionError):
        return None


def last_number(text: str) -> Fraction | None:
    """Parse the last number-like token in free text (a deliberately lenient fallback)."""
    tokens = _NUMBER_TOKEN.findall(text)
    for token in reversed(tokens):
        value = parse_number(token)
        if value is not None:
            return value
    return None


def canonical_number(value: Fraction) -> str:
    """Integer if integral, exact decimal if terminating, otherwise a reduced fraction."""
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    places = max(twos, fives)
    scaled = abs(value) * 10**places
    integer, fractional = divmod(int(scaled), 10**places)
    sign = "-" if value < 0 else ""
    return f"{sign}{integer}.{fractional:0{places}d}".rstrip("0").rstrip(".")


_BINARY_OPS: dict[type[ast.operator], Callable[[Fraction, Fraction], Fraction]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
}
_COMPARISONS: dict[type[ast.cmpop], Callable[[Fraction, Fraction], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCTIONS: dict[str, Callable[..., Fraction]] = {
    "abs": lambda x: abs(x),
    "min": lambda *xs: min(xs),
    "max": lambda *xs: max(xs),
}


class _Evaluator:
    def __init__(self, variables: Mapping[str, Fraction], allow_conditions: bool) -> None:
        self.variables = variables
        self.allow_conditions = allow_conditions

    def number(self, node: ast.AST) -> Fraction:
        value = self.visit(node)
        if isinstance(value, bool):
            raise ExpressionError("expected a number, got a boolean")
        return value

    def boolean(self, node: ast.AST) -> bool:
        value = self.visit(node)
        if not isinstance(value, bool):
            raise ExpressionError("expected a condition, got a number")
        return value

    def visit(self, node: ast.AST) -> Fraction | bool:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise ExpressionError(f"unsupported literal {node.value!r}")
            return _bounded(Fraction(repr(node.value)))
        if isinstance(node, ast.Name):
            if node.id not in self.variables:
                raise ExpressionError(f"unknown name {node.id!r}")
            return self.variables[node.id]
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -self.number(node.operand)
            if isinstance(node.op, ast.UAdd):
                return self.number(node.operand)
            if isinstance(node.op, ast.Not) and self.allow_conditions:
                return not self.boolean(node.operand)
            raise ExpressionError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if self.allow_conditions and isinstance(node, ast.Compare):
            left = self.number(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                compare = _COMPARISONS.get(type(op))
                if compare is None:
                    raise ExpressionError("unsupported comparison")
                right = self.number(comparator)
                if not compare(left, right):
                    return False
                left = right
            return True
        if self.allow_conditions and isinstance(node, ast.BoolOp):
            values = (self.boolean(value) for value in node.values)
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise ExpressionError(f"unsupported syntax: {type(node).__name__}")

    def _binary(self, node: ast.BinOp) -> Fraction:
        left, right = self.number(node.left), self.number(node.right)
        simple = _BINARY_OPS.get(type(node.op))
        if simple is not None:
            return _bounded(simple(left, right))
        if isinstance(node.op, ast.Div | ast.FloorDiv | ast.Mod):
            if right == 0:
                raise ExpressionError("division by zero")
            if isinstance(node.op, ast.Div):
                return _bounded(left / right)
            if isinstance(node.op, ast.FloorDiv):
                return Fraction(left // right)
            return _bounded(left % right)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right) > MAX_EXPONENT:
                raise ExpressionError("exponent must be an integer with |exponent| <= 12")
            if left == 0 and right < 0:
                raise ExpressionError("zero cannot be raised to a negative power")
            return _bounded(left ** int(right))
        raise ExpressionError("unsupported binary operator")

    def _call(self, node: ast.Call) -> Fraction:
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
            raise ExpressionError("only abs(), min() and max() calls are allowed")
        args = [self.number(arg) for arg in node.args]
        if not args:
            raise ExpressionError(f"{node.func.id}() needs arguments")
        return _FUNCTIONS[node.func.id](*args)


def _bounded(value: Fraction) -> Fraction:
    if abs(value) > MAX_ABS_VALUE:
        raise ExpressionError("value out of bounds")
    return value


def _parse(expression: str) -> ast.Expression:
    if len(expression) > 2_000:
        raise ExpressionError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ExpressionError("expression too large")
    return tree


def evaluate_arithmetic(
    expression: str, variables: Mapping[str, Fraction] | None = None
) -> Fraction:
    """Evaluate an arithmetic expression exactly. Supports ``+ - * / // % **``, ``abs/min/max``."""
    tree = _parse(expression)
    return _Evaluator(variables or {}, allow_conditions=False).number(tree.body)


def evaluate_condition(expression: str, variables: Mapping[str, Fraction]) -> bool:
    """Evaluate a boolean constraint such as ``x + y == 10 and y > z``."""
    tree = _parse(expression)
    return _Evaluator(variables, allow_conditions=True).boolean(tree.body)
