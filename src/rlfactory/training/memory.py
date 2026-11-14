"""Rough GPU memory estimation for planning remote runs.

This is a planning heuristic, not a guarantee: fragmentation, kernels, attention implementation,
framework buffers and vLLM KV-cache sizing all move the real number. The workflow therefore always
runs a short smoke step on the actual GPU before a full run (see docs/training/gpu.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.domain.training import LoRASettings, TrainingMethod

GIB = 1024**3


class ModelShape(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden_size: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    intermediate_size: int = Field(ge=1)
    vocab_size: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    num_key_value_heads: int = Field(ge=1)
    tie_word_embeddings: bool = True
    head_dim: int | None = Field(default=None, ge=1)
    """Explicit attention head size (e.g. Qwen3); defaults to hidden_size / num_attention_heads."""

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> ModelShape:
        """Build from a Hugging Face ``config.json`` mapping (decoder-only transformers)."""
        return cls(
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_hidden_layers"]),
            intermediate_size=int(config["intermediate_size"]),
            vocab_size=int(config["vocab_size"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(
                config.get("num_key_value_heads", config["num_attention_heads"])
            ),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            head_dim=int(config["head_dim"]) if config.get("head_dim") else None,
        )

    @property
    def resolved_head_dim(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    def layer_linear_shapes(self) -> list[tuple[int, int]]:
        h, i = self.hidden_size, self.intermediate_size
        q = self.num_attention_heads * self.resolved_head_dim
        kv = self.num_key_value_heads * self.resolved_head_dim
        return [(h, q), (h, kv), (h, kv), (q, h), (h, i), (h, i), (i, h)]

    @property
    def parameters(self) -> int:
        per_layer = sum(a * b for a, b in self.layer_linear_shapes())
        embeddings = self.vocab_size * self.hidden_size * (1 if self.tie_word_embeddings else 2)
        return per_layer * self.num_layers + embeddings


# Shapes read from each model's config.json (local cache or the Hub, 2026-11-14). Add others the
# same way; never type them from memory.
KNOWN_SHAPES: dict[str, ModelShape] = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": ModelShape(
        hidden_size=576,
        num_layers=30,
        intermediate_size=1536,
        vocab_size=49152,
        num_attention_heads=9,
        num_key_value_heads=3,
    ),
    "Qwen/Qwen2.5-0.5B": ModelShape(
        hidden_size=896,
        num_layers=24,
        intermediate_size=4864,
        vocab_size=151936,
        num_attention_heads=14,
        num_key_value_heads=2,
    ),
    "Qwen/Qwen3-0.6B": ModelShape(
        hidden_size=1024,
        num_layers=28,
        intermediate_size=3072,
        vocab_size=151936,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    ),
    "Qwen/Qwen3-1.7B": ModelShape(
        hidden_size=2048,
        num_layers=28,
        intermediate_size=6144,
        vocab_size=151936,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    ),
}


class MemoryEstimate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parameters_b: float
    trainable_parameters_m: float
    weights_gb: float
    trainable_state_gb: float
    activations_gb: float
    logits_gb: float
    rollout_engine_gb: float
    overhead_gb: float
    total_gb: float
    notes: tuple[str, ...]

    def fits(self, gpu_memory_gb: float, headroom: float = 0.9) -> bool:
        return self.total_gb <= gpu_memory_gb * headroom


def estimate_training_memory(
    shape: ModelShape,
    *,
    method: TrainingMethod,
    lora: LoRASettings,
    batch_size: int,
    sequence_length: int,
    gradient_checkpointing: bool = True,
    precision_bytes: int = 2,
    num_generations: int = 1,
    vllm_colocate_fraction: float = 0.0,
    gpu_memory_gb: float = 0.0,
) -> MemoryEstimate:
    params = shape.parameters
    if lora.enabled:
        per_layer = sum(lora.r * (a + b) for a, b in shape.layer_linear_shapes())
        trainable = per_layer * shape.num_layers
    else:
        trainable = params
    quantized = lora.enabled and lora.quantization == "4bit"
    weight_bytes = 0.55 if quantized else precision_bytes
    # fp32 master weights + grads + Adam moments for trainable params (16 B/param), or for full
    # fine-tuning bf16 grads + fp32 master + Adam moments on top of the bf16 weights (14 B/param).
    state_bytes = 16 if lora.enabled else 14
    sequences = batch_size * (2 if method is TrainingMethod.DPO else 1)
    if method is TrainingMethod.GRPO:
        sequences = max(sequences, batch_size)
    per_token_layer = (2 if gradient_checkpointing else 34) * shape.hidden_size * precision_bytes
    activations = sequences * sequence_length * shape.num_layers * per_token_layer
    logits = sequences * sequence_length * shape.vocab_size * 4
    rollout = vllm_colocate_fraction * gpu_memory_gb * GIB
    subtotal = params * weight_bytes + trainable * state_bytes + activations + logits + rollout
    overhead = 0.15 * subtotal + 0.75 * GIB
    notes = [
        "Heuristic estimate; run the smoke step on the target GPU before a full run.",
        f"activations assume {'gradient checkpointing' if gradient_checkpointing else 'no checkpointing'}",
    ]
    if method is TrainingMethod.GRPO:
        notes.append(
            f"GRPO generation memory for {num_generations} samples/prompt is not modelled beyond the vLLM fraction"
        )
    if quantized:
        notes.append("4-bit weights (QLoRA) need bitsandbytes on a CUDA host")
    return MemoryEstimate(
        parameters_b=round(params / 1e9, 3),
        trainable_parameters_m=round(trainable / 1e6, 2),
        weights_gb=round(params * weight_bytes / GIB, 3),
        trainable_state_gb=round(trainable * state_bytes / GIB, 3),
        activations_gb=round(activations / GIB, 3),
        logits_gb=round(logits / GIB, 3),
        rollout_engine_gb=round(rollout / GIB, 3),
        overhead_gb=round(overhead / GIB, 3),
        total_gb=round((subtotal + overhead) / GIB, 3),
        notes=tuple(notes),
    )
