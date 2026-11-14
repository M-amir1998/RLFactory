"""HTTP policies for OpenAI-compatible servers (vLLM, SGLang, hosted APIs).

Request bodies follow the APIs verified in ``docs/references/post-training-stack.md``:
explicit sampling parameters are always sent (servers otherwise apply the model's
``generation_config.json``), engine-specific extras go at the top level of the JSON body, and the
removed vLLM ``guided_*`` fields are never used.
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal

import httpx
from pydantic import JsonValue

from rlfactory.domain.trajectory import TokenUsage
from rlfactory.errors import PolicyError, PolicyTimeoutError
from rlfactory.policies.base import (
    GenerationRequest,
    GenerationResult,
    approximate_tokens,
    to_chat_messages,
)

MaxTokensField = Literal["max_tokens", "max_completion_tokens"]


class OpenAICompatiblePolicy:
    simulated = False
    supports_top_k = False
    health_path = "/v1/models"

    def __init__(
        self,
        policy_id: str,
        base_url: str,
        model: str,
        *,
        api_key_env: str | None = None,
        checkpoint: str | None = None,
        max_tokens_field: MaxTokensField = "max_tokens",
        extra_body: dict[str, JsonValue] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        connect_timeout_s: float = 10.0,
    ) -> None:
        self.policy_id = policy_id
        self.model: str | None = model
        self.checkpoint = checkpoint
        self.max_tokens_field: MaxTokensField = max_tokens_field
        self.extra_body = dict(extra_body or {})
        for forbidden in ("guided_json", "guided_regex", "guided_choice", "guided_grammar"):
            if forbidden in self.extra_body:
                raise ValueError(f"{forbidden} was removed from vLLM; use structured_outputs")
        headers: dict[str, str] = {}
        if api_key_env:
            # Only the *name* of the environment variable is configured; the key never touches disk.
            key = os.environ.get(api_key_env)
            if not key:
                raise PolicyError(f"environment variable {api_key_env} is not set", retryable=False)
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            transport=transport,
            timeout=httpx.Timeout(120.0, connect=connect_timeout_s),
        )

    def build_body(self, request: GenerationRequest) -> dict[str, Any]:
        config = request.config
        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_chat_messages(request.messages),
            "temperature": config.temperature,
            "top_p": config.top_p,
            self.max_tokens_field: config.max_tokens,
            "seed": request.seed,
            "n": 1,
        }
        if config.stop:
            body["stop"] = list(config.stop)
        if config.top_k is not None:
            if not self.supports_top_k:
                raise PolicyError("this server type does not accept top_k", retryable=False)
            body["top_k"] = config.top_k
        body.update(self.extra_body)
        return body

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json=self.build_body(request),
                timeout=request.config.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise PolicyTimeoutError(
                f"generation timed out after {request.config.timeout_s}s"
            ) from exc
        except httpx.TransportError as exc:
            raise PolicyError(f"transport error: {type(exc).__name__}", retryable=True) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise PolicyError(f"server error HTTP {response.status_code}", retryable=True)
        if response.status_code >= 400:
            raise PolicyError(
                f"request rejected HTTP {response.status_code}: {response.text[:200]}",
                retryable=False,
            )
        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise PolicyError("malformed chat-completions response", retryable=True) from exc
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return GenerationResult(
            text=text,
            finish_reason=choice.get("finish_reason"),
            usage=TokenUsage(
                prompt_tokens=int(prompt_tokens or 0),
                completion_tokens=int(
                    completion_tokens if completion_tokens is not None else approximate_tokens(text)
                ),
                estimated=completion_tokens is None,
            ),
            latency_s=time.perf_counter() - started,
            model=data.get("model", self.model),
        )

    async def health(self) -> bool:
        try:
            response = await self._client.get(self.health_path, timeout=10.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()


class VLLMPolicy(OpenAICompatiblePolicy):
    """vLLM >= 0.12 OpenAI-compatible server (verified against 0.33)."""

    supports_top_k = True
    health_path = "/health"

    def __init__(self, policy_id: str, base_url: str, model: str, **kwargs: Any) -> None:
        kwargs.setdefault("max_tokens_field", "max_completion_tokens")
        super().__init__(policy_id, base_url, model, **kwargs)


class SGLangPolicy(OpenAICompatiblePolicy):
    """SGLang OpenAI-compatible server (verified against 0.5.23)."""

    supports_top_k = True
    health_path = "/health"

    async def model_info(self) -> dict[str, Any]:
        response = await self._client.get("/model_info", timeout=10.0)
        response.raise_for_status()
        info: dict[str, Any] = response.json()
        return info
