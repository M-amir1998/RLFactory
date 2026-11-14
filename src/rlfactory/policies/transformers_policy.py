"""Local Hugging Face Transformers policy (CPU smoke inference or single-GPU evaluation).

Imports are lazy so the core package never requires torch. Loading defaults are conservative:
``trust_remote_code`` follows the model reference (False by default), weights must be safetensors,
and ``local_files_only=True`` prevents surprise downloads (see the checkpoint guard).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from rlfactory.domain.model import ModelReference
from rlfactory.domain.trajectory import TokenUsage
from rlfactory.errors import BackendUnavailableError, PolicyError
from rlfactory.policies.base import GenerationRequest, GenerationResult, to_chat_messages


class TransformersPolicy:
    simulated = False

    def __init__(
        self,
        policy_id: str,
        model: ModelReference,
        *,
        local_files_only: bool = True,
        device: str = "cpu",
        checkpoint: str | None = None,
    ) -> None:
        self.policy_id = policy_id
        self.reference = model
        self.model: str | None = model.name
        self.checkpoint = checkpoint
        self.local_files_only = local_files_only
        self.device = device
        self._lock = threading.Lock()
        self._loaded: tuple[Any, Any] | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._loaded is not None:
            return self._loaded
        try:
            import torch  # noqa: F401  (imported for its side effects on transformers)
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendUnavailableError(
                "install the 'trl' extra to use TransformersPolicy"
            ) from exc
        kwargs: dict[str, Any] = {
            "revision": self.reference.revision,
            "trust_remote_code": self.reference.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.reference.name, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            self.reference.name,
            dtype=self.reference.dtype,
            use_safetensors=True,
            **kwargs,
        )
        model.to(self.device)
        model.eval()
        self._loaded = (tokenizer, model)
        return self._loaded

    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        import torch

        started = time.perf_counter()
        with self._lock:
            tokenizer, model = self._load()
            prompt = tokenizer.apply_chat_template(
                to_chat_messages(request.messages), add_generation_prompt=True, tokenize=False
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            config = request.config
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": config.max_tokens,
                "do_sample": config.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if config.temperature > 0:
                generation_kwargs.update(temperature=config.temperature, top_p=config.top_p)
                if config.top_k is not None:
                    generation_kwargs["top_k"] = config.top_k
            torch.manual_seed(request.seed)
            with torch.no_grad():
                output = model.generate(**inputs, **generation_kwargs)
            prompt_length = int(inputs["input_ids"].shape[1])
            completion_ids = output[0][prompt_length:]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        finish = "length" if len(completion_ids) >= config.max_tokens else "stop"
        return GenerationResult(
            text=text,
            finish_reason=finish,
            usage=TokenUsage(prompt_tokens=prompt_length, completion_tokens=len(completion_ids)),
            latency_s=time.perf_counter() - started,
            model=self.model,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            return await asyncio.to_thread(self._generate_sync, request)
        except (BackendUnavailableError, PolicyError):
            raise
        except OSError as exc:
            raise PolicyError(f"model not available locally: {exc}", retryable=False) from exc

    async def aclose(self) -> None:
        self._loaded = None
