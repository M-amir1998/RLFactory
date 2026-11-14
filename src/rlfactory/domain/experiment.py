"""Run manifest: everything needed to reproduce (or audit) an experiment run."""

from __future__ import annotations

from datetime import datetime

from rlfactory.domain.common import CodeVersion, ComponentRef, DomainModel
from rlfactory.domain.model import GenerationConfig


class HardwareInfo(DomainModel):
    platform: str
    python: str
    cpu_count: int
    ram_gb: float
    gpu: str | None = None
    gpu_memory_gb: float | None = None


class SuiteRef(DomainModel):
    name: str
    version: str
    num_tasks: int
    fingerprint: str


class PolicyDescriptor(DomainModel):
    policy_id: str
    kind: str
    model: str | None = None
    checkpoint: str | None = None


class RunManifest(DomainModel):
    run_id: str
    experiment: str
    created_at: datetime
    code: CodeVersion
    seed: int
    task_suites: tuple[SuiteRef, ...]
    environments: tuple[ComponentRef, ...]
    verifiers: tuple[ComponentRef, ...]
    rewards: tuple[ComponentRef, ...]
    policy: PolicyDescriptor
    generation: GenerationConfig | None
    training_backend: str | None
    hardware: HardwareInfo
    config_fingerprint: str
    simulated: bool
    """True when any stage used the mock policy or mock training backend."""
