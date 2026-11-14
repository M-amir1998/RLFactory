from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from rlfactory.datasets.adapters import to_trl_grpo
from rlfactory.datasets.builders import RLPromptDatasetBuilder
from rlfactory.datasets.io import write_dataset
from rlfactory.domain.common import CodeVersion
from rlfactory.domain.model import ModelReference
from rlfactory.domain.task import TaskDomain
from rlfactory.domain.training import (
    DatasetReference,
    DPOParams,
    GRPOParams,
    LaunchMode,
    LoRASettings,
    RewardFunctionSpec,
    SFTParams,
    TrainingConfig,
    TrainingHandle,
    TrainingMethod,
    TrainingRun,
    TrainingState,
)
from rlfactory.environments.registry import default_environment_registry
from rlfactory.errors import (
    BackendUnavailableError,
    ConfigError,
    RegistryError,
    TrainingBackendError,
)
from rlfactory.telemetry.logging import redact
from rlfactory.training.base import (
    EXIT_STATUS_FILE,
    default_backend_registry,
    load_prepared,
    read_metrics,
    render_launch_script,
    save_prepared,
    status_from_workdir,
)
from rlfactory.training.memory import KNOWN_SHAPES, ModelShape, estimate_training_memory
from rlfactory.training.mock.backend import (
    MockTrainingBackend,
    MockTrainingSettings,
    load_mock_checkpoint,
    simulated_profile,
)
from rlfactory.training.openrlhf.backend import OpenRLHFTrainingBackend
from rlfactory.training.trl.backend import TRLBackendSettings, TRLTrainingBackend
from rlfactory.training.trl.config_mapping import (
    REMOVED_KEYS,
    adapt_to_installed,
    build_config_kwargs,
    check_grpo_batching,
)
from rlfactory.training.verl.backend import VerlTrainingBackend
from tests.helpers import tasks_for

MODEL = ModelReference(name="Qwen/Qwen3-0.6B")


def _dataset(tmp_path: Path, kind: str = "sft", records: int = 40) -> DatasetReference:
    schema = {
        "sft": "trl.sft.conversational.prompt_completion",
        "dpo": "trl.dpo.conversational.preference_explicit_prompt",
        "grpo": "trl.grpo.conversational.prompt_only",
    }[kind]
    directory = tmp_path / f"ds-{kind}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (directory / "records.jsonl").write_text(
        "\n".join(
            json.dumps({"lineage": {"domain": "python" if i % 2 else "arithmetic"}})
            for i in range(records)
        )
        + "\n",
        encoding="utf-8",
    )
    return DatasetReference(
        name=f"d-{kind}",
        path=str(directory / "train.jsonl"),
        schema_name=schema,
        num_records=records,
        fingerprint="f",
    )


def _run(
    dataset: DatasetReference, method: TrainingMethod = TrainingMethod.SFT, **config: object
) -> TrainingRun:
    params = {
        TrainingMethod.SFT: SFTParams(),
        TrainingMethod.DPO: DPOParams(),
        TrainingMethod.GRPO: GRPOParams(num_generations=4),
    }[method]
    return TrainingRun(
        run_id=f"run-{method.value}",
        method=method,
        model=MODEL,
        dataset=dataset,
        config=TrainingConfig(method_params=params, **config),
    )  # type: ignore[arg-type]


def test_mock_backend_full_lifecycle(tmp_path: Path) -> None:
    backend = MockTrainingBackend(tmp_path)
    prepared = backend.prepare(
        _run(_dataset(tmp_path), max_steps=10, logging_steps=2, save_steps=5)
    )
    status = backend.inspect(backend.launch(prepared))
    assert status.state is TrainingState.SUCCEEDED and status.step == 10 and status.simulated
    assert [c.step for c in status.checkpoints] == [5, 10]
    metrics = read_metrics(Path(prepared.workdir))
    assert [m.step for m in metrics] == [2, 4, 6, 8, 10]
    assert (
        metrics[0].loss is not None
        and metrics[-1].loss is not None
        and metrics[-1].loss < metrics[0].loss
    )
    checkpoint = load_mock_checkpoint(Path(status.checkpoints[-1].path))
    assert (
        checkpoint.metadata.checkpoint.simulated
        and checkpoint.metadata.framework == "rlfactory-mock"
    )
    summary = json.loads((Path(prepared.workdir) / "training_summary.json").read_text())
    assert summary["simulated"] is True and summary["state"] == "succeeded"


def test_mock_backend_failure_and_resume(tmp_path: Path) -> None:
    backend = MockTrainingBackend(tmp_path, MockTrainingSettings(simulate_failure_at_step=7))
    prepared = backend.prepare(_run(_dataset(tmp_path), max_steps=10, save_steps=5))
    failed = backend.inspect(backend.launch(prepared))
    assert failed.state is TrainingState.FAILED and "simulated failure" in (failed.error or "")
    assert [c.step for c in failed.checkpoints] == [5]
    resumed = backend.inspect(backend.launch(prepared, resume=True))
    assert resumed.state is TrainingState.SUCCEEDED and resumed.step == 10
    summary = json.loads((Path(prepared.workdir) / "training_summary.json").read_text())
    assert summary["resumed_from_step"] == 5


def test_simulated_profile_is_monotone_and_bounded(tmp_path: Path) -> None:
    run = _run(_dataset(tmp_path, records=200))
    settings = MockTrainingSettings()
    profile = simulated_profile(run, settings)
    for domain in TaskDomain:
        assert settings.base_profile.p_correct(domain) <= profile.p_correct(domain) <= 0.98
    assert profile.malformed_probability <= settings.base_profile.malformed_probability


def test_trl_config_mapping_uses_current_names() -> None:
    dataset = DatasetReference(name="d", path="p", schema_name="s", num_records=1, fingerprint="f")
    sft = build_config_kwargs(_run(dataset, warmup_ratio=0.05), output_dir="out")
    assert sft["max_length"] == 1024 and sft["warmup_steps"] == 0.05 and "warmup_ratio" not in sft
    assert sft["model_init_kwargs"] == {"dtype": "bfloat16", "trust_remote_code": False}
    assert sft["report_to"] == "none" and sft["completion_only_loss"] is True
    dpo = build_config_kwargs(_run(dataset, TrainingMethod.DPO), output_dir="out")
    assert dpo["loss_type"] == ["sigmoid"] and "max_prompt_length" not in dpo
    grpo = build_config_kwargs(_run(dataset, TrainingMethod.GRPO), output_dir="out")
    assert (
        grpo["reward_weights"] == [1.0]
        and grpo["vllm_mode"] == "colocate"
        and "max_prompt_length" not in grpo
    )
    cpu = build_config_kwargs(_run(dataset), output_dir="out", cpu=True)
    assert cpu["use_cpu"] and not cpu["bf16"] and cpu["model_init_kwargs"]["dtype"] == "float32"
    for removed in REMOVED_KEYS:
        with pytest.raises(ConfigError, match=removed):
            build_config_kwargs(_run(dataset, backend_overrides={removed: 1}), output_dir="out")


def test_deprecated_keys_are_renamed_only_where_the_new_field_exists() -> None:
    kwargs = {"gradient_checkpointing": True, "learning_rate": 1e-5}
    new = adapt_to_installed(kwargs, {"activation_checkpointing", "gradient_checkpointing"})
    assert new == {"activation_checkpointing": True, "learning_rate": 1e-5}
    # Older Transformers releases have only the old field: the key must survive untouched.
    assert adapt_to_installed(kwargs, {"gradient_checkpointing"}) == kwargs
    explicit = {"gradient_checkpointing": True, "activation_checkpointing": False}
    assert adapt_to_installed(explicit, {"activation_checkpointing"}) == explicit
    assert kwargs == {"gradient_checkpointing": True, "learning_rate": 1e-5}


def test_grpo_server_mode_and_batching_checks() -> None:
    dataset = DatasetReference(name="d", path="p", schema_name="s", num_records=1, fingerprint="f")
    server = TrainingRun(
        run_id="r",
        method=TrainingMethod.GRPO,
        model=MODEL,
        dataset=dataset,
        config=TrainingConfig(method_params=GRPOParams(use_vllm=True, vllm_mode="server")),
    )
    with pytest.raises(ConfigError, match="vllm_server_base_url"):
        build_config_kwargs(server, output_dir="o")
    bad = TrainingRun(
        run_id="r",
        method=TrainingMethod.GRPO,
        model=MODEL,
        dataset=dataset,
        config=TrainingConfig(
            per_device_train_batch_size=3,
            gradient_accumulation_steps=1,
            method_params=GRPOParams(num_generations=4),
        ),
    )
    with pytest.raises(ConfigError, match="divide"):
        check_grpo_batching(bad, num_processes=1)


def test_trl_backend_prepares_self_contained_run(tmp_path: Path) -> None:
    backend = TRLTrainingBackend(tmp_path / "runs")
    prepared = save_prepared(backend.prepare(_run(_dataset(tmp_path))))
    workdir = Path(prepared.workdir)
    assert (workdir / "data" / "train" / "train.jsonl").exists()
    script = (workdir / "launch.sh").read_text()
    assert "accelerate launch" in script and "-m rlfactory.training.trl.entrypoint" in script
    assert "exit_status.json" in script
    assert redact(script) == script and "HF_TOKEN" not in script and "API_KEY" not in script
    spec = json.loads((workdir / "run_config.json").read_text())
    assert spec["config_class"] == "SFTConfig" and spec["train_file"] == "data/train/train.jsonl"
    assert load_prepared(workdir) == prepared
    assert any("not pinned" in note for note in prepared.notes)


def test_trl_backend_guards(tmp_path: Path) -> None:
    backend = TRLTrainingBackend(tmp_path)
    with pytest.raises(ConfigError, match="needs a"):
        backend.prepare(_run(_dataset(tmp_path, "dpo")))
    remote = _run(_dataset(tmp_path)).model_copy(update={"launch_mode": LaunchMode.REMOTE})
    with pytest.raises(TrainingBackendError, match="pinned"):
        backend.prepare(remote)
    assert TRLTrainingBackend(tmp_path, TRLBackendSettings(allow_unpinned_revision=True)).prepare(
        remote
    )
    prepared = backend.prepare(
        _run(_dataset(tmp_path)).model_copy(
            update={
                "launch_mode": LaunchMode.REMOTE,
                "model": MODEL.model_copy(update={"revision": "a" * 40}),
            }
        )
    )
    with pytest.raises(TrainingBackendError, match="remote"):
        backend.launch(prepared)


def _grpo_dataset(tmp_path: Path) -> DatasetReference:
    records = (
        RLPromptDatasetBuilder(environments=default_environment_registry())
        .build(tasks_for(TaskDomain.ARITHMETIC))
        .records
    )
    card = write_dataset(
        tmp_path / "grpo",
        name="g",
        kind="grpo",
        records=records,
        rows=[to_trl_grpo(r) for r in records],
        builder="b",
        builder_config=None,
        build_report={},
        validation={"ok": True},
        source_run_ids=[],
        contamination_counts={},
        code=CodeVersion(rlfactory_version="t"),
    )
    return card.reference(tmp_path / "grpo")


def test_verl_backend_prepares_parquet_reward_shim_and_command(tmp_path: Path) -> None:
    run = TrainingRun(
        run_id="verl-run",
        method=TrainingMethod.GRPO,
        model=MODEL,
        dataset=_grpo_dataset(tmp_path),
        config=TrainingConfig(method_params=GRPOParams(num_generations=8, beta=0.001)),
    )
    prepared = VerlTrainingBackend(tmp_path / "runs").prepare(run)
    workdir = Path(prepared.workdir)
    table = pq.read_table(workdir / "data" / "train.parquet")
    assert set(table.column_names) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }
    assert "compute_score" in (workdir / "reward_fn.py").read_text()
    script = (workdir / "launch.sh").read_text()
    for key in (
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "actor_rollout_ref.rollout.name=vllm",
        "reward.custom_reward_function.path=reward_fn.py",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.model.lora_rank=16",
    ):
        assert key in script
    assert "custom_reward_function.path" not in script.replace(
        "reward.custom_reward_function.path", ""
    )
    with pytest.raises(BackendUnavailableError):
        VerlTrainingBackend(tmp_path).launch(prepared)
    with pytest.raises(ConfigError, match="GRPO"):
        VerlTrainingBackend(tmp_path).prepare(_run(_dataset(tmp_path)))


def test_openrlhf_backend_uses_dotted_flags(tmp_path: Path) -> None:
    run = TrainingRun(
        run_id="orl",
        method=TrainingMethod.GRPO,
        model=MODEL,
        dataset=_grpo_dataset(tmp_path),
        config=TrainingConfig(method_params=GRPOParams(num_generations=4, beta=0.01)),
    )
    prepared = OpenRLHFTrainingBackend(tmp_path / "runs").prepare(run)
    script = (Path(prepared.workdir) / "launch.sh").read_text()
    for flag in (
        "--algo.advantage.estimator group_norm",
        "--reward.remote_url reward_func.py",
        "--actor.model_name_or_path",
        "--rollout.n_samples_per_prompt 4",
        "--algo.kl.use_loss",
    ):
        assert flag in script
    for legacy in ("--pretrain ", "--remote_rm_url", "--advantage_estimator"):
        assert legacy not in script
    rows = (Path(prepared.workdir) / "data" / "prompts.jsonl").read_text().splitlines()
    assert set(json.loads(rows[0])) == {"input", "label"}


def test_status_from_exit_file_and_launch_script(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    handle = TrainingHandle(
        handle_id="h",
        backend="verl",
        run_id="r",
        workdir=str(tmp_path),
        launch_mode=LaunchMode.REMOTE,
        started_at=datetime.now(UTC),
    )
    assert status_from_workdir(handle).state is TrainingState.PREPARED
    (tmp_path / EXIT_STATUS_FILE).write_text('{"exit_code": 3}', encoding="utf-8")
    assert status_from_workdir(handle).state is TrainingState.FAILED
    script = render_launch_script(["python", "-m", "x", "--flag", "a b"], {"HF_HOME": "/cache dir"})
    assert "export HF_HOME='/cache dir'" in script and "'a b'" in script


def test_backend_registry() -> None:
    registry = default_backend_registry()
    assert registry.names() == ["mock", "openrlhf", "trl", "verl"]
    with pytest.raises(RegistryError):
        registry.create("deepspeed-chat", Path())


def test_memory_estimator_shapes_and_methods() -> None:
    smol = KNOWN_SHAPES["HuggingFaceTB/SmolLM2-135M-Instruct"]
    assert 130e6 < smol.parameters < 140e6
    qwen = KNOWN_SHAPES["Qwen/Qwen2.5-0.5B"]
    assert 0.45e9 < qwen.parameters < 0.52e9
    qwen3 = KNOWN_SHAPES["Qwen/Qwen3-0.6B"]  # explicit head_dim=128 != hidden/heads
    assert qwen3.resolved_head_dim == 128 and 0.58e9 < qwen3.parameters < 0.61e9
    assert 1.6e9 < KNOWN_SHAPES["Qwen/Qwen3-1.7B"].parameters < 1.8e9
    lora = estimate_training_memory(
        qwen, method=TrainingMethod.SFT, lora=LoRASettings(), batch_size=4, sequence_length=1024
    )
    full = estimate_training_memory(
        qwen,
        method=TrainingMethod.SFT,
        lora=LoRASettings(enabled=False),
        batch_size=4,
        sequence_length=1024,
    )
    qlora = estimate_training_memory(
        qwen,
        method=TrainingMethod.SFT,
        lora=LoRASettings(quantization="4bit"),
        batch_size=4,
        sequence_length=1024,
    )
    dpo = estimate_training_memory(
        qwen, method=TrainingMethod.DPO, lora=LoRASettings(), batch_size=4, sequence_length=1024
    )
    assert qlora.total_gb < lora.total_gb < full.total_gb
    assert dpo.activations_gb > lora.activations_gb
    assert lora.fits(80) and not full.fits(1)
    shape = ModelShape.from_hf_config(
        {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "vocab_size": 10,
            "num_attention_heads": 2,
        }
    )
    assert shape.num_key_value_heads == 2 and not shape.tie_word_embeddings


def test_grpo_reward_bridge_exposes_components() -> None:
    from rlfactory.training.trl.rewards import build_reward_functions, completion_text

    task = tasks_for(TaskDomain.ARITHMETIC)[0]
    functions = build_reward_functions(
        [
            RewardFunctionSpec(component="correctness"),
            RewardFunctionSpec(component="format", weight=0.1),
        ]
    )
    assert [f.__name__ for f in functions] == ["rlfactory_correctness", "rlfactory_format"]
    completions = [
        [{"role": "assistant", "content": f"<answer>{task.payload.answer}</answer>"}],
        "no answer",
    ]  # type: ignore[union-attr]
    kwargs = {
        "rlfactory_task": [task.model_dump_json()] * 2,
        "rlfactory_reward": ["correctness_format"] * 2,
    }
    assert functions[0](prompts=None, completions=completions, **kwargs) == [1.0, 0.0]
    assert functions[1](prompts=None, completions=completions, **kwargs) == [0.1, 0.0]
    tool_task = tasks_for(TaskDomain.TOOL_USE)[0]
    assert functions[0](completions=["x"], rlfactory_task=[tool_task.model_dump_json()]) == [None]
    with pytest.raises(ValueError, match="rlfactory_task"):
        functions[0](completions=["x"])
    with pytest.raises(ValueError, match="duplicate"):
        build_reward_functions(
            [RewardFunctionSpec(component="format"), RewardFunctionSpec(component="format")]
        )
    assert (
        completion_text([{"content": "a"}, {"content": "b"}]) == "b" and completion_text(5) == "5"
    )


def test_verl_and_openrlhf_reward_shims() -> None:
    from rlfactory.training.openrlhf.reward import response_from_query, score_batch
    from rlfactory.training.verl.reward import compute_score

    task = tasks_for(TaskDomain.ARITHMETIC)[1]
    good = f"<answer>{task.payload.answer}</answer>"  # type: ignore[union-attr]
    result = compute_score(
        "rlfactory/arithmetic", good, task.model_dump_json(), {"reward": "binary"}
    )
    assert result["score"] == 1.0 and result["verified"] == 1.0 and "reward_correctness" in result
    rewards, verified = score_batch(
        ["PROMPT" + good, "PROMPT<answer>-9</answer>"],
        ["PROMPT", "PROMPT"],
        [task.model_dump_json()] * 2,
    )
    assert rewards == [1.0, 0.0] and verified == [1.0, 0.0]
    assert response_from_query("abc", "x") == "abc"
