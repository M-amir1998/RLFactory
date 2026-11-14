"""Optional LLM-judge verifier. Never the primary signal; always recorded with full provenance.

Provenance stored on every result: judge model, prompt-template hash, temperature. Calibrate a
judge against deterministic or human labels (``rlfactory.verifiers.calibration``) before trusting
it for anything that cannot be verified programmatically.
"""

from __future__ import annotations

import os
import re
from typing import Literal, Protocol

import httpx
from typing_extensions import override

from rlfactory.domain.common import stable_hash
from rlfactory.domain.task import Task
from rlfactory.domain.trajectory import Trajectory
from rlfactory.domain.verification import ReasonCode, VerificationResult, VerifierKind
from rlfactory.errors import VerifierError
from rlfactory.verifiers.base import BaseVerifier, reference_answer

DEFAULT_JUDGE_TEMPLATE = """You are grading a model response for correctness.

Task:
{task_prompt}

Reference answer (may be empty):
{reference}

Response to grade:
{response}

Reply with exactly one line: "VERDICT: PASS" or "VERDICT: FAIL"."""

_VERDICT = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)


class JudgeClient(Protocol):
    model: str

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str: ...


class OpenAICompatibleJudgeClient:
    """Synchronous chat-completions client (judges run inside verifier worker threads)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str | None = None,
        timeout_s: float = 60.0,
        max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.max_tokens_field = max_tokens_field
        headers = {}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s, headers=headers, transport=transport
        )

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            self.max_tokens_field: max_tokens,
        }
        response = self._client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"] or "")

    def close(self) -> None:
        self._client.close()


class LLMJudgeVerifier(BaseVerifier):
    kind = VerifierKind.MODEL_JUDGE

    def __init__(
        self,
        name: str,
        client: JudgeClient,
        *,
        template: str = DEFAULT_JUDGE_TEMPLATE,
        temperature: float = 0.0,
        max_tokens: int = 16,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        for field in ("{task_prompt}", "{response}"):
            if field not in template:
                raise ValueError(f"judge template must contain {field}")
        self.client = client
        self.template = template
        self.template_hash = stable_hash(template, length=12)
        self.temperature = temperature
        self.max_tokens = max_tokens

    @override
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        prompt = self.template.format(
            task_prompt=task.prompt,
            reference=reference_answer(task) or "",
            response=trajectory.completion_text,
        )
        provenance: dict[str, str | float] = {
            "judge_model": self.client.model,
            "prompt_template_hash": self.template_hash,
            "temperature": self.temperature,
        }
        try:
            reply = self.client.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise VerifierError(f"judge call failed: {exc}") from exc
        match = _VERDICT.search(reply)
        if match is None:
            return self.fail(ReasonCode.JUDGE_ERROR, raw_reply=reply[:200], **provenance)
        if match.group(1).upper() == "PASS":
            return self.ok(**provenance)
        return self.fail(ReasonCode.JUDGE_REJECTED, **provenance)
