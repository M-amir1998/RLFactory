"""Config-driven sandbox selection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from rlfactory.errors import ConfigError
from rlfactory.sandbox.base import ResourceLimits, Sandbox
from rlfactory.sandbox.docker_sandbox import DEFAULT_IMAGE, DockerSandbox
from rlfactory.sandbox.mock import MockSandbox
from rlfactory.sandbox.subprocess_sandbox import SubprocessSandbox


class SandboxConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["subprocess", "docker", "auto", "mock"] = "subprocess"
    """``auto`` prefers Docker when the daemon and image are available."""
    docker_image: str = DEFAULT_IMAGE
    docker_cpus: float = 1.0
    limits: ResourceLimits = ResourceLimits()


def create_sandbox(config: SandboxConfig) -> Sandbox:
    if config.kind == "subprocess":
        return SubprocessSandbox()
    if config.kind == "mock":
        return MockSandbox()
    docker = DockerSandbox(image=config.docker_image, cpus=config.docker_cpus)
    if config.kind == "docker":
        if not docker.available():
            raise ConfigError(
                "sandbox kind 'docker' requested but the Docker daemon is unavailable"
            )
        if not docker.image_present():
            raise ConfigError(
                f"sandbox image {config.docker_image!r} not present; pull it explicitly first"
            )
        return docker
    if docker.available() and docker.image_present():
        return docker
    return SubprocessSandbox()
