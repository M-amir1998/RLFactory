"""TRL training entrypoint, executed inside a prepared run directory.

``accelerate launch -m rlfactory.training.trl.entrypoint --run-config run_config.json`` on a GPU
host, or ``python -m rlfactory.training.trl.entrypoint --run-config run_config.json`` for a tiny CPU
smoke run. All heavy imports happen inside :func:`main`.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlfactory.config.loader import write_json
from rlfactory.domain.model import CheckpointMetadata, CheckpointReference
from rlfactory.domain.training import (
    CheckpointEvent,
    TrainingMethod,
    TrainingMetricsPoint,
    TrainingState,
    TrainingStatus,
    TrainingSummary,
)
from rlfactory.runtime.git import detect_code_version
from rlfactory.training.base import SUMMARY_FILE, append_metrics, write_status
from rlfactory.training.trl.spec import TRLRunSpec

GIB = 1024**3


def _peak_vram_gb(torch: Any) -> float | None:
    if torch.cuda.is_available():
        return round(float(torch.cuda.max_memory_allocated()) / GIB, 3)
    return None


def _metrics_callback(workdir: Path, handle_id: str, started: float, torch: Any) -> Any:
    from transformers import TrainerCallback

    checkpoints: list[CheckpointEvent] = []

    class RLFactoryMetricsCallback(TrainerCallback):  # type: ignore[misc]
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
            logs = logs or {}
            extra = {
                str(k): float(v)
                for k, v in logs.items()
                if isinstance(v, int | float) and k not in {"loss", "learning_rate"}
            }
            tokens = logs.get(
                "num_input_tokens_seen", getattr(state, "num_input_tokens_seen", None)
            )
            append_metrics(
                workdir,
                TrainingMetricsPoint(
                    step=int(state.global_step),
                    loss=logs.get("loss"),
                    learning_rate=logs.get("learning_rate"),
                    tokens_seen=int(tokens) if tokens else None,
                    wall_time_s=time.perf_counter() - started,
                    peak_vram_gb=_peak_vram_gb(torch),
                    extra=extra,
                ),
            )

        def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            checkpoints.append(
                CheckpointEvent(
                    step=int(state.global_step), path=str(path), created_at=datetime.now(UTC)
                )
            )
            write_status(
                workdir,
                TrainingStatus(
                    handle_id=handle_id,
                    state=TrainingState.RUNNING,
                    step=int(state.global_step),
                    total_steps=int(state.max_steps) or None,
                    checkpoints=tuple(checkpoints),
                ),
            )

    callback = RLFactoryMetricsCallback()
    callback.checkpoints = checkpoints
    return callback


def _peft_config(spec: TRLRunSpec) -> Any:
    lora = spec.run.config.lora
    if not lora.enabled:
        return None
    from peft import LoraConfig

    target = (
        lora.target_modules if isinstance(lora.target_modules, str) else list(lora.target_modules)
    )
    if spec.run.method is TrainingMethod.REWARD_MODEL:
        return LoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=target,
            task_type="SEQ_CLS",
            modules_to_save=["score"],
        )
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=target,
        task_type="CAUSAL_LM",
    )


def _quantization_config(spec: TRLRunSpec, torch: Any) -> Any:
    if not spec.run.config.lora.enabled or spec.run.config.lora.quantization != "4bit":
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def run(spec: TRLRunSpec, workdir: Path) -> TrainingSummary:
    import datasets
    import torch
    import trl

    from rlfactory.training.trl.config_mapping import adapt_to_installed
    from rlfactory.training.trl.rewards import build_reward_functions

    handle_id = f"trl-{spec.run.run_id}"
    started = time.perf_counter()
    write_status(workdir, TrainingStatus(handle_id=handle_id, state=TrainingState.RUNNING))
    train = datasets.load_dataset("json", data_files=str(workdir / spec.train_file), split="train")
    evaluation = (
        datasets.load_dataset("json", data_files=str(workdir / spec.eval_file), split="train")
        if spec.eval_file
        else None
    )
    config_class = getattr(trl, spec.config_class)
    fields = {field.name for field in dataclasses.fields(config_class)}
    config = config_class(**adapt_to_installed(spec.config_kwargs, fields))
    callback = _metrics_callback(workdir, handle_id, started, torch)
    common: dict[str, Any] = {
        "model": spec.run.model.name,
        "args": config,
        "train_dataset": train,
        "eval_dataset": evaluation,
        "peft_config": _peft_config(spec),
        "quantization_config": _quantization_config(spec, torch),
        "callbacks": [callback],
    }
    if spec.run.method is TrainingMethod.GRPO:
        common["reward_funcs"] = build_reward_functions(spec.reward_functions)
    trainer = getattr(trl, spec.trainer_class)(**common)
    trainer.train()
    final_dir = Path(spec.config_kwargs["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    state = trainer.state
    final_loss = next(
        (entry["loss"] for entry in reversed(state.log_history) if "loss" in entry), None
    )
    metadata = CheckpointMetadata(
        checkpoint=CheckpointReference(
            checkpoint_id=f"{spec.run.run_id}-final",
            base_model=spec.run.model,
            adapter_uri=str(final_dir),
            method=spec.run.method.value,
            training_run_id=spec.run.run_id,
            step=int(state.global_step),
        ),
        training_method=spec.run.method.value,
        dataset=spec.run.dataset.name,
        dataset_fingerprint=spec.run.dataset.fingerprint,
        git_sha=detect_code_version().git_sha,
        training_config=spec.config_kwargs,
        framework="trl",
        framework_version=importlib.metadata.version("trl"),
        metrics={"final_loss": float(final_loss)} if final_loss is not None else {},
        created_at=datetime.now(UTC),
    )
    write_json(metadata, final_dir / "rlfactory_checkpoint.json")
    tokens = getattr(state, "num_input_tokens_seen", None)
    summary = TrainingSummary(
        run_id=spec.run.run_id,
        backend="trl",
        method=spec.run.method,
        state=TrainingState.SUCCEEDED,
        steps=int(state.global_step),
        final_loss=float(final_loss) if final_loss is not None else None,
        wall_time_s=round(time.perf_counter() - started, 2),
        tokens_seen=int(tokens) if tokens else None,
        peak_vram_gb=_peak_vram_gb(torch),
        checkpoints=(
            *callback.checkpoints,
            CheckpointEvent(
                step=int(state.global_step), path=str(final_dir), created_at=datetime.now(UTC)
            ),
        ),
        framework_version=importlib.metadata.version("trl"),
        dataset=spec.run.dataset,
        model=spec.run.model,
        simulated=False,
    )
    write_json(summary, workdir / SUMMARY_FILE)
    write_status(
        workdir,
        TrainingStatus(
            handle_id=handle_id,
            state=TrainingState.SUCCEEDED,
            step=summary.steps,
            total_steps=summary.steps,
            checkpoints=summary.checkpoints,
        ),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RLFactory TRL training entrypoint")
    parser.add_argument("--run-config", required=True, type=Path)
    args = parser.parse_args(argv)
    workdir = args.run_config.resolve().parent
    spec = TRLRunSpec.model_validate_json(args.run_config.read_text(encoding="utf-8"))
    try:
        run(spec, workdir)
    except Exception as exc:
        write_status(
            workdir,
            TrainingStatus(
                handle_id=f"trl-{spec.run.run_id}",
                state=TrainingState.FAILED,
                error=f"{type(exc).__name__}: {exc}"[:1000],
            ),
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
