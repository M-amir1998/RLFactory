"""Rollout configuration and retry policy."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.domain.model import GenerationConfig
from rlfactory.domain.task import TaskDomain
from rlfactory.domain.trajectory import FailureKind
from rlfactory.trajectories.store import RetentionPolicy

DEFAULT_RETRY_ON: tuple[FailureKind, ...] = (
    FailureKind.POLICY_TIMEOUT,
    FailureKind.POLICY_ERROR,
    FailureKind.SANDBOX_ERROR,
    FailureKind.INFRASTRUCTURE_ERROR,
)


class RetryPolicy(BaseModel):
    """Retries apply to infrastructure failures only.

    Model failures (``invalid_output``, ``sandbox_timeout``) are legitimate outcomes and are never
    retried; ``verifier_error`` is not retried by default because deterministic verifiers fail
    deterministically.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_initial_s: float = Field(default=0.5, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    backoff_max_s: float = Field(default=10.0, ge=0.0)
    retry_on: tuple[FailureKind, ...] = DEFAULT_RETRY_ON

    def delay(self, attempt: int) -> float:
        """Backoff before attempt ``attempt + 1`` (attempts are 1-based)."""
        return min(
            self.backoff_max_s, self.backoff_initial_s * self.backoff_multiplier ** (attempt - 1)
        )


class RolloutConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    samples_per_task: int = Field(default=4, ge=1, le=256)
    concurrency: int = Field(default=4, ge=1, le=256)
    generation: GenerationConfig = GenerationConfig(temperature=0.8, max_tokens=512)
    seed: int = 0
    reward: str = "binary"
    verifier_overrides: dict[TaskDomain, str] = Field(default_factory=dict)
    retry: RetryPolicy = RetryPolicy()
    episode_timeout_s: float = Field(default=600.0, gt=0.0)
    flush_every: int = Field(default=64, ge=1)
    retention: RetentionPolicy = RetentionPolicy.ALL
    retention_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    resume: bool = True
    """Skip trajectory IDs that already have a COMPLETED record in the store."""
