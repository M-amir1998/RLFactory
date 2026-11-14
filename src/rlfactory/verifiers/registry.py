"""Config-driven verifier registry.

Verifiers are referenced by name from tasks. Each name maps to a typed spec; the spec hash becomes
part of the verifier version so trajectories record exactly which configuration scored them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.domain.common import stable_hash
from rlfactory.domain.verification import VerifierKind
from rlfactory.errors import ConfigError, RegistryError
from rlfactory.verifiers.base import Verifier
from rlfactory.verifiers.composite import CompositeComponent, CompositeVerifier
from rlfactory.verifiers.judge import (
    DEFAULT_JUDGE_TEMPLATE,
    JudgeClient,
    LLMJudgeVerifier,
    OpenAICompatibleJudgeClient,
)
from rlfactory.verifiers.matching import ExactMatchVerifier, NumericVerifier, RegexVerifier
from rlfactory.verifiers.structured import JSONSchemaVerifier, ToolOutcomeVerifier, UnitTestVerifier


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExactMatchSpec(_Spec):
    type: Literal["exact_match"] = "exact_match"
    case_sensitive: bool = True
    normalize_whitespace: bool = True


class NumericSpec(_Spec):
    type: Literal["numeric"] = "numeric"
    rel_tol: float = Field(default=1e-9, ge=0.0)
    abs_tol: float = Field(default=1e-9, ge=0.0)
    answer_source: Literal["parsed", "parsed_or_last_number"] = "parsed"


class JSONSchemaSpec(_Spec):
    type: Literal["json_schema"] = "json_schema"
    check_expected: bool = True
    check_constraints: bool = True


class UnitTestSpec(_Spec):
    type: Literal["unit_test"] = "unit_test"
    partial_credit: bool = True


class RegexSpec(_Spec):
    type: Literal["regex"] = "regex"
    pattern: str
    target: Literal["completion", "answer"] = "completion"
    mode: Literal["search", "fullmatch"] = "search"
    kind: VerifierKind = VerifierKind.HEURISTIC


class ToolOutcomeSpec(_Spec):
    type: Literal["tool_outcome"] = "tool_outcome"
    require_tools: bool = True
    rel_tol: float = Field(default=1e-9, ge=0.0)


class CompositeComponentSpec(_Spec):
    verifier: str
    role: Literal["correctness", "format", "constraint"]
    weight: float = Field(gt=0.0)
    required: bool = False


class CompositeSpec(_Spec):
    type: Literal["composite"] = "composite"
    components: tuple[CompositeComponentSpec, ...]
    min_correctness_share: float = Field(default=0.5, ge=0.0, le=1.0)


class JudgeSpec(_Spec):
    type: Literal["llm_judge"] = "llm_judge"
    base_url: str
    model: str
    api_key_env: str | None = None
    template: str | None = None
    temperature: float = 0.0
    max_tokens: int = 16
    timeout_s: float = 60.0


VerifierSpec = Annotated[
    ExactMatchSpec
    | NumericSpec
    | JSONSchemaSpec
    | UnitTestSpec
    | RegexSpec
    | ToolOutcomeSpec
    | CompositeSpec
    | JudgeSpec,
    Field(discriminator="type"),
]


class VerifierConfig(BaseModel):
    """``configs/verifiers/*.yaml``: named verifier specs (merged over the defaults)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    verifiers: dict[str, VerifierSpec] = Field(default_factory=dict)


DEFAULT_VERIFIER_SPECS: dict[str, VerifierSpec] = {
    "exact_match": ExactMatchSpec(),
    "exact_strict": ExactMatchSpec(normalize_whitespace=False),
    "numeric": NumericSpec(),
    "numeric_lenient": NumericSpec(answer_source="parsed_or_last_number"),
    "json_schema": JSONSchemaSpec(),
    "unit_test": UnitTestSpec(),
    "unit_test_binary": UnitTestSpec(partial_credit=False),
    "tool_outcome": ToolOutcomeSpec(),
    "answer_tag_format": RegexSpec(
        pattern=r"<answer>.*?</answer>", kind=VerifierKind.DETERMINISTIC
    ),
    "python_block_format": RegexSpec(
        pattern=r"```python\s*\n.*?```", kind=VerifierKind.DETERMINISTIC
    ),
    "arithmetic_composite": CompositeSpec(
        components=(
            CompositeComponentSpec(
                verifier="numeric", role="correctness", weight=0.9, required=True
            ),
            CompositeComponentSpec(verifier="answer_tag_format", role="format", weight=0.1),
        )
    ),
    "python_composite": CompositeSpec(
        components=(
            CompositeComponentSpec(
                verifier="unit_test", role="correctness", weight=0.9, required=True
            ),
            CompositeComponentSpec(verifier="python_block_format", role="format", weight=0.1),
        )
    ),
}

JudgeClientFactory = Callable[[JudgeSpec], JudgeClient]


def _default_judge_client(spec: JudgeSpec) -> JudgeClient:
    return OpenAICompatibleJudgeClient(
        spec.base_url, spec.model, api_key_env=spec.api_key_env, timeout_s=spec.timeout_s
    )


class VerifierRegistry:
    """Lazily instantiates verifiers from specs; composite references resolve recursively."""

    def __init__(
        self,
        specs: Mapping[str, VerifierSpec] | None = None,
        *,
        judge_client_factory: JudgeClientFactory = _default_judge_client,
    ) -> None:
        self._specs: dict[str, VerifierSpec] = dict(DEFAULT_VERIFIER_SPECS)
        if specs:
            self._specs.update(specs)
        self._instances: dict[str, Verifier] = {}
        self._judge_client_factory = judge_client_factory

    def names(self) -> list[str]:
        return sorted(self._specs)

    def spec(self, name: str) -> VerifierSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise RegistryError(
                f"unknown verifier {name!r}; known: {', '.join(self.names())}"
            ) from exc

    def register(self, name: str, verifier: Verifier) -> None:
        """Register a pre-built verifier instance (e.g. a custom or test verifier)."""
        self._instances[name] = verifier

    def get(self, name: str) -> Verifier:
        if name not in self._instances:
            self._instances[name] = self._build(name, ())
        return self._instances[name]

    def _build(self, name: str, stack: tuple[str, ...]) -> Verifier:
        if name in stack:
            raise ConfigError(f"verifier reference cycle: {' -> '.join((*stack, name))}")
        if name in self._instances:
            return self._instances[name]
        spec = self.spec(name)
        fingerprint = stable_hash(spec.model_dump(mode="json"), length=8)
        if isinstance(spec, ExactMatchSpec):
            return ExactMatchVerifier(
                name,
                case_sensitive=spec.case_sensitive,
                normalize_whitespace=spec.normalize_whitespace,
                config_fingerprint=fingerprint,
            )
        if isinstance(spec, NumericSpec):
            return NumericVerifier(
                name,
                rel_tol=spec.rel_tol,
                abs_tol=spec.abs_tol,
                answer_source=spec.answer_source,
                config_fingerprint=fingerprint,
            )
        if isinstance(spec, JSONSchemaSpec):
            return JSONSchemaVerifier(
                name,
                check_expected=spec.check_expected,
                check_constraints=spec.check_constraints,
                config_fingerprint=fingerprint,
            )
        if isinstance(spec, UnitTestSpec):
            return UnitTestVerifier(
                name, partial_credit=spec.partial_credit, config_fingerprint=fingerprint
            )
        if isinstance(spec, RegexSpec):
            return RegexVerifier(
                name,
                spec.pattern,
                target=spec.target,
                mode=spec.mode,
                kind=spec.kind,
                config_fingerprint=fingerprint,
            )
        if isinstance(spec, ToolOutcomeSpec):
            return ToolOutcomeVerifier(
                name,
                require_tools=spec.require_tools,
                rel_tol=spec.rel_tol,
                config_fingerprint=fingerprint,
            )
        if isinstance(spec, CompositeSpec):
            components = [
                CompositeComponent(
                    verifier=self._build(component.verifier, (*stack, name)),
                    role=component.role,
                    weight=component.weight,
                    required=component.required,
                )
                for component in spec.components
            ]
            return CompositeVerifier(
                name,
                components,
                min_correctness_share=spec.min_correctness_share,
                config_fingerprint=fingerprint,
            )
        client = self._judge_client_factory(spec)
        return LLMJudgeVerifier(
            name,
            client,
            template=spec.template or DEFAULT_JUDGE_TEMPLATE,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            config_fingerprint=fingerprint,
        )
