"""Reference implementations for Python tasks.

Their source (via ``inspect.getsource``) becomes each task's reference solution, and they compute
the expected outputs of hidden tests. Keep them stdlib-only, non-recursive and docstring-free.
"""

from __future__ import annotations


def sum_of_evens(values: list[int]) -> int:
    return sum(v for v in values if v % 2 == 0)


def count_vowels(text: str) -> int:
    return sum(1 for ch in text.lower() if ch in "aeiou")


def running_max(values: list[int]) -> list[int]:
    result: list[int] = []
    for v in values:
        result.append(v if not result or v > result[-1] else result[-1])
    return result


def dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def second_largest(values: list[int]) -> int | None:
    distinct = sorted(set(values), reverse=True)
    return distinct[1] if len(distinct) >= 2 else None


def is_palindrome(text: str) -> bool:
    cleaned = [ch.lower() for ch in text if ch.isalnum()]
    return cleaned == cleaned[::-1]


def rotate_left(values: list[int], k: int) -> list[int]:
    if not values:
        return []
    k %= len(values)
    return values[k:] + values[:k]


def char_frequency(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            key = ch.lower()
            counts[key] = counts.get(key, 0) + 1
    return counts
