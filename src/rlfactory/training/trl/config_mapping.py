"""Map a canonical :class:`TrainingRun` onto TRL 1.18 config keyword arguments.

Field names and defaults were verified against the installed TRL 1.18.0 dataclasses (see
docs/references/post-training-stack.md); ``tests/integration/test_trl_compat.py`` re-checks every
emitted key against the installed config classes whenever the ``trl`` extra is present.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from rlfactory.domain.training import (
    DPOParams,
    GRPOParams,
    RewardModelParams,
    SFTParams,
    TrainingMethod,
    TrainingRun,
)
from rlfactory.errors import ConfigError

TARGET_TRL_VERSION = "1.18.0"

CONFIG_CLASSES: dict[TrainingMethod, str] = {
    TrainingMethod.SFT: "SFTConfig",
    TrainingMethod.DPO: "DPOConfig",
    TrainingMethod.GRPO: "GRPOConfig",
    TrainingMethod.REWARD_MODEL: "RewardConfig",
}
TRAINER_CLASSES: dict[TrainingMethod, str] = {
    TrainingMethod.SFT: "SFTTrainer",
    TrainingMethod.DPO: "DPOTrainer",
    TrainingMethod.GRPO: "GRPOTrainer",
    TrainingMethod.REWARD_MODEL: "RewardTrainer",
}
EXPECTED_SCHEMAS: dict[TrainingMethod, str] = {
    TrainingMethod.SFT: "trl.sft.conversational.prompt_completion",
    TrainingMethod.DPO: "trl.dpo.conversational.preference_explicit_prompt",
    TrainingMethod.GRPO: "trl.grpo.conversational.prompt_only",
    TrainingMethod.REWARD_MODEL: "trl.reward.conversational.preference_implicit_prompt",
}

# Keys that older TRL/Transformers accepted but 1.18 / 5.x removed or deprecated.
REMOVED_KEYS: dict[str, str] = {
    "max_seq_length": "renamed to max_length in TRL 0.16",
    "max_prompt_length": "removed from DPO and GRPO; filter long prompts when building the dataset",
    "tokenizer": "pass processing_class instead (TRL 0.12+)",
    "warmup_ratio": "removed in Transformers 5; RLFactory maps warmup_ratio to warmup_steps",
    "use_liger_kernel": "deprecated in TRL 1.15; use model_init_kwargs={'use_kernels': True}",
    "torch_dtype": "use dtype (Transformers 4.56+)",
}

# Keys Transformers renamed while keeping a deprecated alias (5.x mirrors the new value back onto
# the old attribute, so trainer code reading the old name keeps working; the alias goes in v6).
# Run specs carry the old spelling, which every supported release accepts; the entrypoint renames
# it on the training host when the installed config class has the new field.
RENAMED_KEYS: dict[str, str] = {
    "gradient_checkpointing": "activation_checkpointing",
    "gradient_checkpointing_kwargs": "activation_checkpointing_kwargs",
}


def adapt_to_installed(kwargs: Mapping[str, Any], field_names: Collection[str]) -> dict[str, Any]:
    """Rename deprecated keys for the config class installed where training actually runs."""
    adapted = dict(kwargs)
    for old, new in RENAMED_KEYS.items():
        if old in adapted and new in field_names and new not in adapted:
            adapted[new] = adapted.pop(old)
    return adapted


def build_config_kwargs(run: TrainingRun, *, output_dir: str, cpu: bool = False) -> dict[str, Any]:
    config = run.config
    model = run.model
    for key, reason in REMOVED_KEYS.items():
        if key in config.backend_overrides:
            raise ConfigError(
                f"backend override {key!r} is not valid for TRL {TARGET_TRL_VERSION}: {reason}"
            )
    dtype = "float32" if cpu else ("bfloat16" if config.bf16 else model.dtype)
    model_init: dict[str, Any] = {"dtype": dtype, "trust_remote_code": model.trust_remote_code}
    if model.revision is not None:
        model_init["revision"] = model.revision
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing and not cpu,
        "bf16": config.bf16 and not cpu,
        "use_cpu": cpu,
        "warmup_steps": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "seed": run.seed,
        "report_to": "none",
        "include_num_input_tokens_seen": "non_padding",
        "trust_remote_code": model.trust_remote_code,
        "model_init_kwargs": model_init,
    }
    params = config.method_params
    if isinstance(params, SFTParams):
        kwargs.update(
            max_length=params.max_length,
            packing=params.packing,
            completion_only_loss=params.completion_only_loss,
        )
    elif isinstance(params, DPOParams):
        kwargs.update(
            beta=params.beta,
            loss_type=list(params.loss_type),
            max_length=params.max_length,
            precompute_ref_log_probs=params.precompute_ref_log_probs,
        )
    elif isinstance(params, GRPOParams):
        kwargs.update(
            num_generations=params.num_generations,
            max_completion_length=params.max_completion_length,
            temperature=params.temperature,
            top_p=params.top_p,
            beta=params.beta,
            loss_type=params.loss_type,
            scale_rewards=params.scale_rewards,
            num_iterations=params.num_iterations,
            epsilon=params.epsilon,
            mask_truncated_completions=params.mask_truncated_completions,
            log_completions=params.log_completions,
            reward_weights=[spec.weight for spec in params.reward_functions],
            use_vllm=params.use_vllm and not cpu,
            vllm_mode=params.vllm_mode,
        )
        if params.use_vllm and params.vllm_mode == "server":
            if not params.vllm_server_base_url:
                raise ConfigError("GRPO vllm_mode='server' needs vllm_server_base_url")
            kwargs["vllm_server_base_url"] = params.vllm_server_base_url
        elif params.use_vllm:
            kwargs["vllm_gpu_memory_utilization"] = params.vllm_gpu_memory_utilization
    elif isinstance(params, RewardModelParams):
        kwargs.update(max_length=params.max_length)
        if params.center_rewards_coefficient is not None:
            kwargs["center_rewards_coefficient"] = params.center_rewards_coefficient
    kwargs.update(config.backend_overrides)
    return kwargs


def check_grpo_batching(run: TrainingRun, num_processes: int) -> None:
    """TRL requires num_generations to divide the generation batch (per-device x processes x
    gradient-accumulation steps, the default ``steps_per_generation``)."""
    params = run.config.method_params
    if not isinstance(params, GRPOParams):
        return
    batch = (
        run.config.per_device_train_batch_size
        * num_processes
        * run.config.gradient_accumulation_steps
    )
    if batch % params.num_generations:
        raise ConfigError(
            f"GRPO num_generations={params.num_generations} must divide per_device_train_batch_size x "
            f"num_processes x gradient_accumulation_steps = {batch}"
        )
