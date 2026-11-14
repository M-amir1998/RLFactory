"""Convert canonical records to framework dataset formats.

TRL 1.18 formats (verified, see docs/references/post-training-stack.md):

* SFT: conversational prompt-completion -> ``{"prompt": [...], "completion": [...]}``
  (completion-only loss is TRL's default for this format);
* DPO: conversational preference with explicit prompt -> ``{"prompt", "chosen", "rejected"}``;
* GRPO: conversational prompt-only -> ``{"prompt": [...]}`` plus extra columns, which TRL passes to
  reward functions as keyword arguments (``rlfactory_task`` carries the full task for verification);
* Reward model: conversational preference with implicit prompt -> ``{"chosen", "rejected"}``.

Only columns the trainer needs are emitted; RLFactory metadata stays in ``records.jsonl``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from rlfactory.datasets.records import DatasetMessage, PreferenceRecord, RLPromptRecord, SFTRecord

DatasetKind = Literal["sft", "dpo", "grpo", "reward_model"]

TRL_SCHEMAS: dict[DatasetKind, str] = {
    "sft": "trl.sft.conversational.prompt_completion",
    "dpo": "trl.dpo.conversational.preference_explicit_prompt",
    "grpo": "trl.grpo.conversational.prompt_only",
    "reward_model": "trl.reward.conversational.preference_implicit_prompt",
}

REQUIRED_COLUMNS: dict[DatasetKind, tuple[str, ...]] = {
    "sft": ("prompt", "completion"),
    "dpo": ("prompt", "chosen", "rejected"),
    "grpo": ("prompt", "rlfactory_task", "rlfactory_reward"),
    "reward_model": ("chosen", "rejected"),
}


def _messages(messages: Sequence[DatasetMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def to_trl_sft(record: SFTRecord) -> dict[str, Any]:
    return {"prompt": _messages(record.prompt), "completion": _messages(record.completion)}


def to_trl_dpo(record: PreferenceRecord) -> dict[str, Any]:
    return {
        "prompt": _messages(record.prompt),
        "chosen": _messages(record.chosen),
        "rejected": _messages(record.rejected),
    }


def to_trl_grpo(record: RLPromptRecord) -> dict[str, Any]:
    return {
        "prompt": _messages(record.prompt),
        "rlfactory_task": record.task.model_dump_json(),
        "rlfactory_reward": record.reward,
    }


def to_trl_reward(record: PreferenceRecord) -> dict[str, Any]:
    prompt = _messages(record.prompt)
    return {
        "chosen": prompt + _messages(record.chosen),
        "rejected": prompt + _messages(record.rejected),
    }


def to_verl_rl(record: RLPromptRecord, index: int) -> dict[str, Any]:
    """verl 0.9 RL parquet row: data_source, prompt, ability, reward_model, extra_info."""
    return {
        "data_source": f"rlfactory/{record.task.domain.value}",
        "prompt": _messages(record.prompt),
        "ability": record.task.domain.value,
        "reward_model": {"style": "rule", "ground_truth": record.task.model_dump_json()},
        "extra_info": {"index": index, "task_id": record.task.task_id, "reward": record.reward},
    }


def to_openrlhf_rl(record: RLPromptRecord) -> dict[str, Any]:
    """OpenRLHF prompt dataset row (``--data.input_key input --data.label_key label``)."""
    return {"input": _messages(record.prompt), "label": record.task.model_dump_json()}
