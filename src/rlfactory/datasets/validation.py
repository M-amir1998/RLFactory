"""Dataset validation for canonical records and TRL-formatted JSONL files."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlfactory.datasets.adapters import REQUIRED_COLUMNS, DatasetKind
from rlfactory.datasets.records import DatasetMessage, PreferenceRecord, RLPromptRecord, SFTRecord
from rlfactory.tasks.contamination import ContaminationFilter

_VALID_ROLES = {"system", "user", "assistant"}


@dataclass
class DatasetValidationReport:
    kind: str
    records: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicate_prompts: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "records": self.records,
            "ok": self.ok,
            "errors": self.errors[:50],
            "error_count": len(self.errors),
            "warnings": self.warnings[:50],
            "duplicate_prompts": self.duplicate_prompts,
        }


def _check_prompt(prompt: Sequence[DatasetMessage], where: str, errors: list[str]) -> None:
    if not any(m.role == "user" for m in prompt):
        errors.append(f"{where}: prompt has no user message")
    if prompt[-1].role != "user":
        errors.append(f"{where}: prompt must end with a user message (got {prompt[-1].role})")
    if any(m.role == "system" for m in prompt[1:]):
        errors.append(f"{where}: system message only allowed first")


def _check_completion(messages: Sequence[DatasetMessage], where: str, errors: list[str]) -> None:
    if messages[-1].role != "assistant":
        errors.append(f"{where}: completion must end with an assistant message")
    if not messages[-1].content.strip():
        errors.append(f"{where}: empty completion")


def _finite(value: float | None) -> bool:
    return value is None or math.isfinite(value)


def _prompt_key(prompt: Sequence[DatasetMessage]) -> str:
    return json.dumps([(m.role, m.content) for m in prompt])


def validate_sft(
    records: Sequence[SFTRecord], contamination: ContaminationFilter | None = None
) -> DatasetValidationReport:
    report = DatasetValidationReport("sft", len(records))
    for record in records:
        where = record.record_id
        _check_prompt(record.prompt, where, report.errors)
        _check_completion(record.completion, where, report.errors)
        if not _finite(record.reward):
            report.errors.append(f"{where}: non-finite reward")
        if contamination is not None and record.lineage.task_id in contamination.task_ids:
            report.errors.append(
                f"{where}: evaluation task {record.lineage.task_id} leaked into training data"
            )
    counts = Counter(_prompt_key(r.prompt) for r in records)
    report.duplicate_prompts = sum(c - 1 for c in counts.values() if c > 1)
    if report.duplicate_prompts:
        report.warnings.append(
            f"{report.duplicate_prompts} duplicate prompts (expected with diverse/all selection)"
        )
    return report


def validate_preference(
    records: Sequence[PreferenceRecord], contamination: ContaminationFilter | None = None
) -> DatasetValidationReport:
    report = DatasetValidationReport("preference", len(records))
    for record in records:
        where = record.record_id
        _check_prompt(record.prompt, where, report.errors)
        _check_completion(record.chosen, f"{where}.chosen", report.errors)
        _check_completion(record.rejected, f"{where}.rejected", report.errors)
        if record.chosen == record.rejected:
            report.errors.append(f"{where}: chosen == rejected")
        if not (_finite(record.chosen_reward) and _finite(record.rejected_reward)):
            report.errors.append(f"{where}: non-finite reward metadata")
        if (
            record.chosen_reward is not None
            and record.rejected_reward is not None
            and record.chosen_reward < record.rejected_reward
        ):
            report.warnings.append(f"{where}: chosen has lower reward than rejected")
        if contamination is not None and record.lineage.task_id in contamination.task_ids:
            report.errors.append(
                f"{where}: evaluation task {record.lineage.task_id} leaked into training data"
            )
    return report


def validate_rl(
    records: Sequence[RLPromptRecord], contamination: ContaminationFilter | None = None
) -> DatasetValidationReport:
    report = DatasetValidationReport("grpo", len(records))
    for record in records:
        _check_prompt(record.prompt, record.record_id, report.errors)
        if contamination is not None and contamination.reason(record.task) is not None:
            report.errors.append(f"{record.record_id}: contaminated task {record.task.task_id}")
    counts = Counter(_prompt_key(r.prompt) for r in records)
    report.duplicate_prompts = sum(c - 1 for c in counts.values() if c > 1)
    if report.duplicate_prompts:
        report.errors.append(f"{report.duplicate_prompts} duplicate prompts in an RL prompt set")
    return report


def _check_messages(value: Any, where: str, errors: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        errors.append(f"{where}: expected a non-empty list of messages")
        return []
    for message in value:
        if not isinstance(message, dict) or message.get("role") not in _VALID_ROLES:
            errors.append(f"{where}: invalid message role {message!r:.60}")
        elif not isinstance(message.get("content"), str):
            errors.append(f"{where}: message content must be a string")
    return [m for m in value if isinstance(m, dict)]


def validate_trl_file(path: Path, kind: DatasetKind) -> DatasetValidationReport:
    """Validate a TRL-formatted JSONL file (as produced by the adapters) independently."""
    report = DatasetValidationReport(kind, 0)
    prompts: Counter[str] = Counter()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        report.records += 1
        where = f"{path.name}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            report.errors.append(f"{where}: invalid JSON ({exc.msg})")
            continue
        missing = [c for c in REQUIRED_COLUMNS[kind] if c not in row]
        if missing:
            report.errors.append(f"{where}: missing columns {missing}")
            continue
        for column in ("prompt", "completion", "chosen", "rejected"):
            if column in row:
                messages = _check_messages(row[column], f"{where}.{column}", report.errors)
                if (
                    column != "prompt"
                    and messages
                    and not str(messages[-1].get("content", "")).strip()
                ):
                    report.errors.append(f"{where}.{column}: empty final message")
        if "chosen" in row and row.get("chosen") == row.get("rejected"):
            report.errors.append(f"{where}: chosen == rejected")
        if "prompt" in row:
            prompts[json.dumps(row["prompt"], sort_keys=True)] += 1
    report.duplicate_prompts = sum(c - 1 for c in prompts.values() if c > 1)
    if kind == "grpo" and report.duplicate_prompts:
        report.errors.append(f"{report.duplicate_prompts} duplicate prompts")
    return report
