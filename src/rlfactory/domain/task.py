"""Task model: a verifiable unit of work with a typed, domain-specific payload."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from rlfactory.domain.common import DomainModel, stable_hash


class TaskDomain(StrEnum):
    ARITHMETIC = "arithmetic"
    STRUCTURED_OUTPUT = "structured_output"
    PYTHON = "python"
    TRANSFORMATION = "transformation"
    TOOL_USE = "tool_use"


class Difficulty(DomainModel):
    """Simple, observable difficulty signals.

    ``level`` is a coarse 1-5 bucket derived from ``signals`` (operation counts, input sizes,
    program complexity, tool steps). It is a sampling aid, not a measure of cognitive difficulty.
    """

    level: int = Field(ge=1, le=5)
    signals: dict[str, float] = Field(default_factory=dict)


class TaskMetadata(DomainModel):
    version: str = "1"
    family: str
    """Template family; used for grouped splits and contamination control."""
    generator: str | None = None
    template: str | None = None
    params: dict[str, JsonValue] = Field(default_factory=dict)
    difficulty: Difficulty | None = None
    tags: tuple[str, ...] = ()
    source: Literal["synthetic", "fixture", "external", "mutation"] = "synthetic"
    parent_task_id: str | None = None
    mutation: str | None = None
    benchmark_id: str | None = None


class ToolCall(DomainModel):
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


# --------------------------------------------------------------------------- payloads


class ArithmeticPayload(DomainModel):
    kind: Literal["arithmetic"] = "arithmetic"
    expression: str
    answer: str
    """Exact canonical answer: an integer (``"17"``) or a reduced fraction (``"-3/4"``)."""


class StructuredOutputPayload(DomainModel):
    kind: Literal["structured_output"] = "structured_output"
    json_schema: dict[str, JsonValue]
    expected: JsonValue = None
    """Exact expected JSON value, or ``None`` when several answers are valid."""
    constraints: tuple[str, ...] = ()
    """Boolean expressions over top-level answer fields (logic tasks)."""
    reference_answer: JsonValue = None
    """One known-valid answer, used for task validation and by the mock policy oracle."""

    @model_validator(mode="after")
    def _has_ground_truth(self) -> StructuredOutputPayload:
        if self.expected is None and not self.constraints:
            raise ValueError("structured tasks need an expected value or constraints")
        if self.expected is None and self.reference_answer is None:
            raise ValueError("constraint tasks need a reference answer for validation")
        return self


class PythonTestCase(DomainModel):
    args: tuple[JsonValue, ...]
    expected: JsonValue


class PythonFunctionPayload(DomainModel):
    kind: Literal["python"] = "python"
    entry_point: str
    signature: str
    tests: tuple[PythonTestCase, ...] = Field(min_length=1)
    """Hidden tests: never rendered into the prompt."""
    public_examples: tuple[PythonTestCase, ...] = ()
    reference_solution: str

    @field_validator("entry_point")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError(f"entry point {value!r} is not a Python identifier")
        return value


class TransformationPayload(DomainModel):
    kind: Literal["transformation"] = "transformation"
    operation: str
    input_text: str
    expected_output: str


class ToolUsePayload(DomainModel):
    kind: Literal["tool_use"] = "tool_use"
    tools: tuple[str, ...] = Field(min_length=1)
    kv_store: dict[str, JsonValue] = Field(default_factory=dict)
    document: JsonValue = None
    expected_answer: str
    required_tools: tuple[str, ...] = ()
    max_steps: int = Field(default=6, ge=1, le=32)
    reference_plan: tuple[ToolCall, ...] = ()
    """A known-good tool-call sequence; used for validation and the mock policy oracle."""


TaskPayload = Annotated[
    ArithmeticPayload
    | StructuredOutputPayload
    | PythonFunctionPayload
    | TransformationPayload
    | ToolUsePayload,
    Field(discriminator="kind"),
]

_PAYLOAD_DOMAIN: dict[str, TaskDomain] = {
    "arithmetic": TaskDomain.ARITHMETIC,
    "structured_output": TaskDomain.STRUCTURED_OUTPUT,
    "python": TaskDomain.PYTHON,
    "transformation": TaskDomain.TRANSFORMATION,
    "tool_use": TaskDomain.TOOL_USE,
}


class Task(DomainModel):
    """Canonical task record.

    ``environment`` and ``verifier`` are registry names, so the same task can be replayed under a
    different verifier configuration (e.g. a verifier-quality study) without rewriting it.
    """

    task_id: str
    domain: TaskDomain
    prompt: str = Field(min_length=1)
    environment: str
    verifier: str
    payload: TaskPayload
    metadata: TaskMetadata

    @model_validator(mode="after")
    def _payload_matches_domain(self) -> Task:
        if _PAYLOAD_DOMAIN[self.payload.kind] is not self.domain:
            raise ValueError(f"payload kind {self.payload.kind!r} does not match {self.domain}")
        return self

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def family(self) -> str:
        return self.metadata.family

    def fingerprint(self) -> str:
        """Content hash covering prompt, payload and version (not the id)."""
        return stable_hash(self.domain, self.prompt, self.payload, self.metadata.version)


def make_task_id(domain: TaskDomain, template: str, params: dict[str, JsonValue]) -> str:
    """Deterministic task id derived from the generating template and parameters."""
    return f"{domain.value}-{template}-{stable_hash(template, params, length=12)}"
