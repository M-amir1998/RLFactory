"""Sandboxed execution of untrusted, model-generated programs.

Generated code is NEVER executed in the RLFactory process. See ``docs/security/sandbox.md`` for
the (limited) guarantees of each implementation.
"""

from rlfactory.sandbox.base import (
    ExecutionRequest,
    ExecutionResult,
    ResourceLimits,
    Sandbox,
    SandboxCapabilities,
)

__all__ = [
    "ExecutionRequest",
    "ExecutionResult",
    "ResourceLimits",
    "Sandbox",
    "SandboxCapabilities",
]
