"""Minimal model-promotion registry: candidate -> validated | rejected (small JSON metadata only)."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.config.loader import write_json
from rlfactory.domain.result import GateReport, PromotionRecord, PromotionState
from rlfactory.errors import RLFactoryError


class PromotionError(RLFactoryError):
    """Invalid promotion transition."""


class _RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current: dict[str, PromotionRecord] = Field(default_factory=dict)
    history: list[PromotionRecord] = Field(default_factory=list)


class PromotionRegistry:
    """Stores the latest state per checkpoint plus an append-only history. Weights never live here."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> _RegistryFile:
        if not self.path.exists():
            return _RegistryFile()
        return _RegistryFile.model_validate(json.loads(self.path.read_text(encoding="utf-8")))

    def _save(self, data: _RegistryFile) -> None:
        write_json(data.model_dump(mode="json"), self.path)

    def get(self, checkpoint_id: str) -> PromotionRecord | None:
        return self._load().current.get(checkpoint_id)

    def all(self) -> list[PromotionRecord]:
        return list(self._load().current.values())

    def register_candidate(self, checkpoint_id: str, *, simulated: bool = False) -> PromotionRecord:
        with self._lock:
            data = self._load()
            existing = data.current.get(checkpoint_id)
            if existing is not None and existing.state is not PromotionState.CANDIDATE:
                raise PromotionError(f"{checkpoint_id} is already {existing.state.value}")
            record = PromotionRecord(
                checkpoint_id=checkpoint_id,
                state=PromotionState.CANDIDATE,
                updated_at=datetime.now(UTC),
                simulated=simulated,
            )
            data.current[checkpoint_id] = record
            data.history.append(record)
            self._save(data)
            return record

    def decide(
        self, checkpoint_id: str, report: GateReport, *, evaluation_id: str | None = None
    ) -> PromotionRecord:
        with self._lock:
            data = self._load()
            existing = data.current.get(checkpoint_id)
            if existing is None:
                raise PromotionError(f"{checkpoint_id} was never registered as a candidate")
            if existing.state is not PromotionState.CANDIDATE:
                raise PromotionError(
                    f"{checkpoint_id} was already decided ({existing.state.value})"
                )
            reasons = tuple(f"{c.rule.name}: {c.detail}" for c in report.checks if not c.passed)
            record = PromotionRecord(
                checkpoint_id=checkpoint_id,
                state=PromotionState.VALIDATED if report.passed else PromotionState.REJECTED,
                updated_at=datetime.now(UTC),
                gate_passed=report.passed,
                reasons=reasons,
                evaluation_id=evaluation_id,
                simulated=existing.simulated,
            )
            data.current[checkpoint_id] = record
            data.history.append(record)
            self._save(data)
            return record
