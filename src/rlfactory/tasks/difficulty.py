"""Coarse difficulty estimation from observable signals.

The level is a sampling aid (e.g. difficulty-balanced curricula), computed from counts such as
reasoning operations, input size, program complexity and tool steps. It is NOT a measure of how hard
a task is for a particular model; use per-task pass rates for that.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from rlfactory.domain.task import Difficulty

_WEIGHTS: Mapping[str, float] = {
    "operations": 0.6,
    "constraints": 0.5,
    "tool_steps": 0.45,
    "tests": 0.1,
    "input_size": 0.12,
}


def estimate_difficulty(signals: Mapping[str, float]) -> Difficulty:
    score = sum(weight * signals.get(name, 0.0) for name, weight in _WEIGHTS.items())
    score += 0.5 * math.log10(1.0 + abs(signals.get("max_operand", 0.0)))
    score += signals.get("program_complexity", 0.0) / 40.0
    level = min(5, max(1, 1 + int(score // 1.5)))
    return Difficulty(level=level, signals={k: float(v) for k, v in sorted(signals.items())})
