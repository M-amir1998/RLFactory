"""Dataset persistence: canonical records + framework-format rows + a dataset card."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from rlfactory.config.loader import read_json_model, write_json
from rlfactory.datasets.adapters import TRL_SCHEMAS, DatasetKind
from rlfactory.domain.common import CodeVersion, stable_hash
from rlfactory.domain.training import DatasetReference
from rlfactory.runtime.disk import DiskGuard, estimate_dataset_bytes

CARD_FILE = "dataset_card.json"
RECORDS_FILE = "records.jsonl"
TRAIN_FILE = "train.jsonl"


class DatasetCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: DatasetKind
    schema_name: str
    num_records: int
    fingerprint: str
    builder: str
    builder_config: dict[str, JsonValue] = Field(default_factory=dict)
    build_report: dict[str, JsonValue] = Field(default_factory=dict)
    validation: dict[str, JsonValue] = Field(default_factory=dict)
    source_run_ids: tuple[str, ...] = ()
    contamination_filter: dict[str, int] = Field(default_factory=dict)
    created_at: datetime
    code: CodeVersion

    def reference(self, directory: Path) -> DatasetReference:
        return DatasetReference(
            name=self.name,
            path=str(directory / TRAIN_FILE),
            schema_name=self.schema_name,
            num_records=self.num_records,
            fingerprint=self.fingerprint,
        )


def _write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_dataset(
    directory: Path,
    *,
    name: str,
    kind: DatasetKind,
    records: Sequence[BaseModel],
    rows: Sequence[dict[str, Any]],
    builder: str,
    builder_config: BaseModel | None,
    build_report: dict[str, Any],
    validation: dict[str, Any],
    source_run_ids: Sequence[str],
    contamination_counts: dict[str, int],
    code: CodeVersion,
    disk_guard: DiskGuard | None = None,
) -> DatasetCard:
    if disk_guard is not None:
        disk_guard.ensure(directory, estimate_dataset_bytes(len(rows)), f"writing dataset {name}")
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl([json.loads(r.model_dump_json()) for r in records], directory / RECORDS_FILE)
    _write_jsonl(rows, directory / TRAIN_FILE)
    card = DatasetCard(
        name=name,
        kind=kind,
        schema_name=TRL_SCHEMAS[kind],
        num_records=len(rows),
        fingerprint=stable_hash(rows, length=16),
        builder=builder,
        builder_config=builder_config.model_dump(mode="json") if builder_config else {},
        build_report=json.loads(json.dumps(build_report, default=str)),
        validation=json.loads(json.dumps(validation, default=str)),
        source_run_ids=tuple(sorted(set(source_run_ids))),
        contamination_filter=contamination_counts,
        created_at=datetime.now(UTC),
        code=code,
    )
    write_json(card, directory / CARD_FILE)
    return card


def read_card(directory: Path) -> DatasetCard:
    return read_json_model(directory / CARD_FILE, DatasetCard)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
