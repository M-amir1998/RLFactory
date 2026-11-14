"""Output-format conventions shared by environments, verifiers, rewards and the mock oracle.

* Final answers: ``<answer>...</answer>`` (the last occurrence wins).
* Python solutions: a fenced ```` ```python ```` block (the last one wins).
* Tool calls: ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from rlfactory.domain.task import ToolCall

ANSWER_TAG = "answer"
TOOL_CALL_TAG = "tool_call"

_CODE_BLOCK = re.compile(r"```(?P<lang>[A-Za-z0-9_+\-]*)[ \t]*\r?\n(?P<body>.*?)```", re.DOTALL)


@lru_cache(maxsize=16)
def _tag_pattern(tag: str) -> re.Pattern[str]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*", tag):
        raise ValueError(f"invalid tag name {tag!r}")
    return re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)


def extract_tag(text: str, tag: str = ANSWER_TAG) -> str | None:
    """Return the stripped content of the last ``<tag>...</tag>`` block, or ``None``."""
    matches = _tag_pattern(tag).findall(text)
    if not matches:
        return None
    return str(matches[-1]).strip()


def has_tag(text: str, tag: str = ANSWER_TAG) -> bool:
    return extract_tag(text, tag) is not None


def extract_code_block(text: str, language: str = "python") -> str | None:
    """Return the last fenced block tagged ``language`` (or the last untagged block)."""
    tagged: list[str] = []
    untagged: list[str] = []
    for match in _CODE_BLOCK.finditer(text):
        lang = match.group("lang").lower()
        if lang == language.lower():
            tagged.append(match.group("body"))
        elif not lang:
            untagged.append(match.group("body"))
    blocks = tagged or untagged
    if not blocks:
        return None
    code = blocks[-1].strip("\n")
    return code if code.strip() else None


def strip_code_fence(text: str) -> str:
    match = _CODE_BLOCK.fullmatch(text.strip())
    return match.group("body").strip() if match else text.strip()


def extract_json_answer(text: str) -> str | None:
    """Return the raw JSON text from ``<answer>`` (fences tolerated) or a ```` ```json ```` block."""
    tagged = extract_tag(text, ANSWER_TAG)
    if tagged is not None:
        return strip_code_fence(tagged)
    return extract_code_block(text, "json")


def parse_tool_call(text: str) -> tuple[ToolCall | None, str | None]:
    """Parse the last ``<tool_call>`` block.

    Returns ``(call, None)`` on success, ``(None, error)`` for a malformed call and
    ``(None, None)`` when the text contains no tool call at all.
    """
    raw = extract_tag(text, TOOL_CALL_TAG)
    if raw is None:
        return None, None
    try:
        data = json.loads(strip_code_fence(raw))
    except json.JSONDecodeError:
        return None, "tool call is not valid JSON"
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None, 'tool call must be an object with a string "name"'
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        return None, '"arguments" must be an object'
    return ToolCall(name=data["name"], arguments=arguments), None


def render_tool_call(call: ToolCall) -> str:
    payload = {"name": call.name, "arguments": call.arguments}
    return f"<{TOOL_CALL_TAG}>{json.dumps(payload, sort_keys=True)}</{TOOL_CALL_TAG}>"
