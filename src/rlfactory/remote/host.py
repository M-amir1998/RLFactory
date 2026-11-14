"""Remote host configuration. Credentials are never stored: SSH uses the user's agent/config."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rlfactory.config.loader import load_model

_SECRET_NAME = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE)", re.IGNORECASE
)


class GPUProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "unknown"
    count: int = Field(default=1, ge=1)
    memory_gb: float = Field(default=24.0, gt=0.0)


class RemoteEnvironment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    setup_commands: tuple[str, ...] = ("uv sync --extra trl --extra gpu",)
    activate: str | None = "source .venv/bin/activate"
    env_vars: dict[str, str] = Field(default_factory=dict)
    """Non-secret variables only (e.g. HF_HOME). Secrets must already exist on the host."""

    @field_validator("env_vars")
    @classmethod
    def _no_secrets(cls, value: dict[str, str]) -> dict[str, str]:
        for name in value:
            if _SECRET_NAME.search(name):
                raise ValueError(
                    f"refusing to store {name!r}: configure secrets on the remote host, not in RLFactory"
                )
        return value


class RemoteHost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    host: str
    """Hostname or an alias from ~/.ssh/config (preferred: keeps users/keys out of RLFactory)."""
    user: str | None = None
    port: int = Field(default=22, ge=1, le=65535)
    workspace: str
    """Absolute directory on the remote host where run directories are synced."""
    code_dir: str
    """Absolute path of the RLFactory checkout on the remote host."""
    gpu: GPUProfile = GPUProfile()
    environment: RemoteEnvironment = RemoteEnvironment()
    connect_timeout_s: int = Field(default=15, ge=1)

    @field_validator("workspace", "code_dir")
    @classmethod
    def _absolute(cls, value: str) -> str:
        if not PurePosixPath(value).is_absolute():
            raise ValueError("remote paths must be absolute POSIX paths")
        return value

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def ssh_options(self) -> list[str]:
        return [
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout_s}",
            "-p",
            str(self.port),
        ]


def load_remote_host(path: Path) -> RemoteHost:
    return load_model(path, RemoteHost)
