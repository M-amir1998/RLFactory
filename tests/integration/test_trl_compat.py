"""TRL compatibility: every key the adapter emits must exist on the installed TRL config classes.

Skipped unless the optional ``trl`` extra is installed (never in the default Jenkins pipeline).
The real CPU smoke run additionally needs a tiny model in the local Hugging Face cache and
``RLFACTORY_RUN_TRL_SMOKE=1``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from rlfactory.domain.model import ModelReference
from rlfactory.domain.training import (
    DatasetReference,
    DPOParams,
    GRPOParams,
    RewardModelParams,
    SFTParams,
    TrainingConfig,
    TrainingMethod,
    TrainingRun,
)
from rlfactory.training.trl.config_mapping import (
    CONFIG_CLASSES,
    TRAINER_CLASSES,
    adapt_to_installed,
    build_config_kwargs,
)

trl = pytest.importorskip("trl", reason="optional 'trl' extra not installed")
pytestmark = pytest.mark.trl

PARAMS = {
    TrainingMethod.SFT: SFTParams(),
    TrainingMethod.DPO: DPOParams(),
    TrainingMethod.GRPO: GRPOParams(num_generations=2),
    TrainingMethod.REWARD_MODEL: RewardModelParams(center_rewards_coefficient=0.01),
}


@pytest.mark.parametrize("method", list(TrainingMethod))
def test_emitted_keys_exist_and_configs_instantiate(method: TrainingMethod, tmp_path: Path) -> None:
    run = TrainingRun(
        run_id="compat",
        method=method,
        model=ModelReference(name="tiny"),
        dataset=DatasetReference(
            name="d", path="p", schema_name="s", num_records=1, fingerprint="f"
        ),
        # GPU-mode mapping (it is what emits gradient checkpointing); bf16 stays off because
        # Transformers validates bf16 hardware support at construction time on this CPU machine.
        config=TrainingConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            bf16=False,
            method_params=PARAMS[method],
        ),
    )
    kwargs = build_config_kwargs(run, output_dir=str(tmp_path), cpu=False)
    config_class = getattr(trl, CONFIG_CLASSES[method])
    fields = {f.name for f in dataclasses.fields(config_class)}
    assert set(kwargs) <= fields, sorted(set(kwargs) - fields)
    with warnings.catch_warnings():
        # A deprecated key would still work today and break on the next major release.
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        config = config_class(**adapt_to_installed(kwargs, fields))
    assert config.output_dir == str(tmp_path)
    assert config.gradient_checkpointing is run.config.gradient_checkpointing
    assert hasattr(trl, TRAINER_CLASSES[method])


@pytest.mark.tiny_model
@pytest.mark.skipif(
    os.environ.get("RLFACTORY_RUN_TRL_SMOKE") != "1", reason="set RLFACTORY_RUN_TRL_SMOKE=1 to run"
)
def test_real_cpu_sft_smoke(tmp_path: Path, repo_root: Path) -> None:
    from rlfactory.runtime.hf_cache import find_cached_model

    if find_cached_model("HuggingFaceTB/SmolLM2-135M-Instruct") is None:
        pytest.skip("SmolLM2-135M-Instruct not in the local Hugging Face cache")
    rows = [
        {
            "prompt": [{"role": "user", "content": f"What is {i}+{i}?"}],
            "completion": [{"role": "assistant", "content": f"<answer>{2 * i}</answer>"}],
        }
        for i in range(8)
    ]
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    from rlfactory.training.trl.backend import TRLBackendSettings, TRLTrainingBackend

    run = TrainingRun(
        run_id="smoke",
        method=TrainingMethod.SFT,
        model=ModelReference(name="HuggingFaceTB/SmolLM2-135M-Instruct"),
        dataset=DatasetReference(
            name="d",
            path=str(data / "train.jsonl"),
            schema_name="trl.sft.conversational.prompt_completion",
            num_records=8,
            fingerprint="f",
        ),
        config=TrainingConfig(
            max_steps=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=1000,
            method_params=SFTParams(max_length=128),
        ),
    )
    prepared = TRLTrainingBackend(tmp_path / "runs", TRLBackendSettings(cpu_smoke=True)).prepare(
        run
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rlfactory.training.trl.entrypoint",
            "--run-config",
            "run_config.json",
        ],
        cwd=prepared.workdir,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    summary = json.loads((Path(prepared.workdir) / "training_summary.json").read_text())
    assert (
        summary["state"] == "succeeded" and summary["steps"] == 2 and summary["simulated"] is False
    )
