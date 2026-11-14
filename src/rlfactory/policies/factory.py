"""Config-driven policy construction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from rlfactory.domain.experiment import PolicyDescriptor
from rlfactory.domain.model import ModelReference
from rlfactory.domain.task import Task
from rlfactory.policies.base import Policy
from rlfactory.policies.http import OpenAICompatiblePolicy, SGLangPolicy, VLLMPolicy
from rlfactory.policies.mock import MockPolicy, MockPolicyProfile
from rlfactory.policies.transformers_policy import TransformersPolicy


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MockPolicySpec(_Spec):
    kind: Literal["mock"] = "mock"
    policy_id: str = "mock-base"
    seed: int = 0
    profile: MockPolicyProfile = MockPolicyProfile()
    checkpoint: str | None = None


class HTTPPolicySpec(_Spec):
    kind: Literal["openai_compatible", "vllm", "sglang"]
    policy_id: str
    base_url: str
    model: str
    api_key_env: str | None = None
    checkpoint: str | None = None
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] | None = None
    extra_body: dict[str, JsonValue] = Field(default_factory=dict)


class TransformersPolicySpec(_Spec):
    kind: Literal["transformers"] = "transformers"
    policy_id: str
    model: ModelReference
    local_files_only: bool = True
    device: str = "cpu"
    checkpoint: str | None = None


PolicySpec = Annotated[
    MockPolicySpec | HTTPPolicySpec | TransformersPolicySpec, Field(discriminator="kind")
]


def describe_policy(spec: PolicySpec) -> PolicyDescriptor:
    if isinstance(spec, MockPolicySpec):
        return PolicyDescriptor(
            policy_id=spec.policy_id,
            kind="mock",
            model="rlfactory/mock-policy",
            checkpoint=spec.checkpoint,
        )
    if isinstance(spec, HTTPPolicySpec):
        return PolicyDescriptor(
            policy_id=spec.policy_id, kind=spec.kind, model=spec.model, checkpoint=spec.checkpoint
        )
    return PolicyDescriptor(
        policy_id=spec.policy_id,
        kind="transformers",
        model=spec.model.describe(),
        checkpoint=spec.checkpoint,
    )


def build_policy(spec: PolicySpec, tasks: Iterable[Task] = ()) -> Policy:
    """``tasks`` feeds the mock policy's oracle; real policies ignore it."""
    if isinstance(spec, MockPolicySpec):
        return MockPolicy(
            tasks,
            spec.profile,
            policy_id=spec.policy_id,
            seed=spec.seed,
            checkpoint=spec.checkpoint,
        )
    if isinstance(spec, HTTPPolicySpec):
        kwargs: dict[str, object] = {
            "api_key_env": spec.api_key_env,
            "checkpoint": spec.checkpoint,
            "extra_body": dict(spec.extra_body),
        }
        if spec.max_tokens_field is not None:
            kwargs["max_tokens_field"] = spec.max_tokens_field
        cls = {
            "openai_compatible": OpenAICompatiblePolicy,
            "vllm": VLLMPolicy,
            "sglang": SGLangPolicy,
        }[spec.kind]
        return cls(spec.policy_id, spec.base_url, spec.model, **kwargs)  # type: ignore[arg-type]
    return TransformersPolicy(
        spec.policy_id,
        spec.model,
        local_files_only=spec.local_files_only,
        device=spec.device,
        checkpoint=spec.checkpoint,
    )
