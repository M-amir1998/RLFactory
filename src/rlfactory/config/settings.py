"""Process-level settings resolved from environment variables (never from secrets files)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config_dir: Path = Path("configs")
    results_dir: Path = Path("results")
    disk_reserve_gb: float = Field(default=5.0, ge=0.0)
    log_format: Literal["text", "json"] = "text"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        values: dict[str, object] = {}
        if env.get("RLFACTORY_CONFIG_DIR"):
            values["config_dir"] = Path(env["RLFACTORY_CONFIG_DIR"])
        if env.get("RLFACTORY_RESULTS_DIR"):
            values["results_dir"] = Path(env["RLFACTORY_RESULTS_DIR"])
        if env.get("RLFACTORY_DISK_RESERVE_GB"):
            values["disk_reserve_gb"] = float(env["RLFACTORY_DISK_RESERVE_GB"])
        if env.get("RLFACTORY_LOG_FORMAT"):
            values["log_format"] = env["RLFACTORY_LOG_FORMAT"]
        if env.get("RLFACTORY_LOG_LEVEL"):
            values["log_level"] = env["RLFACTORY_LOG_LEVEL"]
        return cls.model_validate(values)

    def run_dir(self, run_id: str) -> Path:
        if not run_id or any(sep in run_id for sep in ("/", "\\", "..")):
            raise ValueError(f"invalid run id {run_id!r}")
        return self.results_dir / run_id
