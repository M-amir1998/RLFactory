"""The run specification written to ``run_config.json`` and consumed by the TRL entrypoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from rlfactory.domain.training import RewardFunctionSpec, TrainingRun


class TRLRunSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run: TrainingRun
    trainer_class: str
    config_class: str
    config_kwargs: dict[str, Any]
    train_file: str
    eval_file: str | None = None
    reward_functions: tuple[RewardFunctionSpec, ...] = ()
    cpu: bool = False
    target_trl_version: str
