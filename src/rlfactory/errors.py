"""Exception hierarchy.

Errors carry enough structure for the rollout layer to map them onto the failure taxonomy
(:class:`rlfactory.domain.trajectory.FailureKind`) and decide whether a retry is appropriate.
"""

from __future__ import annotations


class RLFactoryError(Exception):
    """Base class for all RLFactory errors."""


class ConfigError(RLFactoryError):
    """Invalid or inconsistent configuration."""


class RegistryError(RLFactoryError):
    """Unknown component name or duplicate registration."""


class TaskValidationError(RLFactoryError):
    """A task cannot be independently solved/verified by its environment."""


class PolicyError(RLFactoryError):
    """The policy (model server, local model) failed to produce a generation."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class PolicyTimeoutError(PolicyError):
    """The policy did not answer within the generation timeout."""

    def __init__(self, message: str = "policy timed out") -> None:
        super().__init__(message, retryable=True)


class SandboxError(RLFactoryError):
    """The sandbox itself failed (could not start, crashed); not the generated program's fault."""


class EnvironmentRuntimeError(RLFactoryError):
    """An environment failed for infrastructure reasons (not because of the model's action)."""


class VerifierError(RLFactoryError):
    """A verifier crashed; recorded as ``verifier_error`` and never silently scored as zero."""


class StoreError(RLFactoryError):
    """Trajectory store read/write failure."""


class DatasetValidationError(RLFactoryError):
    """A built dataset failed validation."""


class InsufficientDiskError(RLFactoryError):
    """An operation would violate the configured free-disk reserve."""


class TrainingBackendError(RLFactoryError):
    """A training backend could not prepare, launch or inspect a run."""


class BackendUnavailableError(TrainingBackendError):
    """The backend's framework is not installed or not usable in this environment."""


class RemoteExecutionError(RLFactoryError):
    """A remote (SSH) command failed."""


class ExperimentError(RLFactoryError):
    """An experiment DAG is invalid or a stage failed."""
