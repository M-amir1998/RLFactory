from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from rlfactory.domain.model import ModelReference
from rlfactory.domain.training import (
    DatasetReference,
    LaunchMode,
    PreparedTrainingRun,
    TrainingHandle,
    TrainingRun,
    TrainingState,
)
from rlfactory.errors import RemoteExecutionError
from rlfactory.remote.commands import CommandResult, DryRunCommandRunner, LocalCommandRunner
from rlfactory.remote.host import RemoteHost, load_remote_host
from rlfactory.remote.runner import LAUNCH_COMMAND, RemoteTrainingRunner
from rlfactory.remote.ssh import SSHClient
from rlfactory.training.memory import KNOWN_SHAPES, estimate_training_memory

HOST = RemoteHost(
    name="box",
    host="gpu-alias",
    workspace="/srv/runs",
    code_dir="/srv/RLFactory",
    environment={"env_vars": {"HF_HOME": "/srv/hf cache"}},
)
SMI = "NVIDIA L4, 23034, 1024\n"


def _prepared(tmp_path: Path, pinned: bool = True) -> PreparedTrainingRun:
    model = ModelReference(name="Qwen/Qwen2.5-0.5B", revision="a" * 40 if pinned else None)
    run = TrainingRun(
        run_id="r1",
        method="sft",
        model=model,  # type: ignore[arg-type]
        dataset=DatasetReference(
            name="d", path="p", schema_name="s", num_records=1, fingerprint="f"
        ),
        launch_mode=LaunchMode.REMOTE,
    )
    (tmp_path / "run_config.json").write_text(
        json.dumps({"config_kwargs": {"max_steps": -1}}), encoding="utf-8"
    )
    return PreparedTrainingRun(
        run=run, backend="trl", workdir=str(tmp_path), command=("bash", "launch.sh")
    )


def _responder(command: tuple[str, ...]) -> CommandResult | None:
    joined = " ".join(command)
    if "nvidia-smi" in joined:
        return CommandResult(command, 0, SMI)
    if "nohup bash launch.sh" in joined:
        return CommandResult(command, 0, "4242\n")
    return None


def test_host_validation_rejects_secrets_and_relative_paths(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="refusing"):
        RemoteHost(
            name="b",
            host="h",
            workspace="/w",
            code_dir="/c",
            environment={"env_vars": {"HF_TOKEN": "x"}},
        )
    with pytest.raises(ValidationError, match="absolute"):
        RemoteHost(name="b", host="h", workspace="relative", code_dir="/c")
    assert (
        RemoteHost(name="b", host="h", user="ubuntu", workspace="/w", code_dir="/c").target
        == "ubuntu@h"
    )
    path = tmp_path / "host.yaml"
    path.write_text("name: b\nhost: h\nworkspace: /w\ncode_dir: /c\n", encoding="utf-8")
    assert load_remote_host(path).host == "h"


def test_example_host_config_is_valid(repo_root: Path) -> None:
    assert (
        load_remote_host(repo_root / "configs" / "remote" / "gpu-host.example.yaml").gpu.memory_gb
        == 24
    )


def test_ssh_command_construction_quotes_and_batch_mode() -> None:
    runner = DryRunCommandRunner()
    client = SSHClient(HOST, runner)
    client.run(client.in_dir("/srv/runs/r 1", "python -V"))
    command = runner.commands[0]
    assert (
        command[:2] == ("ssh", "-o") and "BatchMode=yes" in command and command[-2] == "gpu-alias"
    )
    assert "cd '/srv/runs/r 1'" in command[-1] and "export HF_HOME='/srv/hf cache'" in command[-1]
    assert command[-1].startswith(
        "cd /srv/RLFactory && source .venv/bin/activate && cd '/srv/runs/r 1'"
    )
    assert "|| true" not in command[-1]  # a failed cd/activation must abort, never fall through


def test_preflight_gpu_memory_and_pinning(tmp_path: Path) -> None:
    runner = DryRunCommandRunner(_responder)
    remote = RemoteTrainingRunner(HOST, SSHClient(HOST, runner))
    shape = KNOWN_SHAPES["Qwen/Qwen2.5-0.5B"]
    from rlfactory.domain.training import LoRASettings, TrainingMethod

    small = estimate_training_memory(
        shape, method=TrainingMethod.SFT, lora=LoRASettings(), batch_size=1, sequence_length=512
    )
    report = remote.preflight(_prepared(tmp_path), small)
    assert report.ok and report.gpus[0].name == "NVIDIA L4"
    huge = small.model_copy(update={"total_gb": 500.0})
    assert not remote.preflight(_prepared(tmp_path), huge).ok
    unpinned = remote.preflight(_prepared(tmp_path, pinned=False), None)
    assert any("not pinned" in p for p in unpinned.problems)
    no_gpu = RemoteTrainingRunner(HOST, SSHClient(HOST, DryRunCommandRunner()))
    assert any("no NVIDIA GPU" in p for p in no_gpu.preflight(_prepared(tmp_path), None).problems)


def test_sync_smoke_launch_status_and_fetch(tmp_path: Path) -> None:
    runner = DryRunCommandRunner(_responder)
    remote = RemoteTrainingRunner(HOST, SSHClient(HOST, runner))
    prepared = _prepared(tmp_path)
    assert remote.sync_run(prepared) == "/srv/runs/r1"
    remote.smoke(prepared, max_steps=2)
    smoke_spec = json.loads((tmp_path / "run_config.smoke.json").read_text())
    assert smoke_spec["config_kwargs"]["max_steps"] == 2
    handle = remote.launch(prepared)
    assert handle.handle_id.endswith("pid4242") and handle.remote_host == "box"
    remote.prepare_environment()
    assert any("uv sync --extra trl --extra gpu" in " ".join(c) for c in runner.commands)
    fetched = remote.fetch_results(handle, tmp_path / "fetched")
    assert "training_summary.json" in fetched and "outputs/final/adapter_config.json" in fetched
    assert not any(".safetensors" in " ".join(c) for c in runner.commands)


def test_status_parsing_and_failures(tmp_path: Path) -> None:
    handle = TrainingHandle(
        handle_id="h",
        backend="trl",
        run_id="r",
        workdir="/srv/runs/r",
        launch_mode=LaunchMode.REMOTE,
        started_at=datetime.now(UTC),
    )

    def finished(command: tuple[str, ...]) -> CommandResult:
        return CommandResult(command, 0, '@@{"exit_code": 0}@@tail')

    remote = RemoteTrainingRunner(HOST, SSHClient(HOST, DryRunCommandRunner(finished)))
    assert remote.status(handle).state is TrainingState.SUCCEEDED
    unreachable = RemoteTrainingRunner(
        HOST, SSHClient(HOST, DryRunCommandRunner(lambda c: CommandResult(c, 255, "", "x")))
    )
    assert unreachable.status(handle).error == "host unreachable"
    with pytest.raises(RemoteExecutionError, match="launch"):
        unreachable.launch(_prepared(tmp_path))


def test_local_command_runner_never_uses_a_shell() -> None:
    result = LocalCommandRunner().run(["python", "-c", "print('ok')"], timeout=30)
    assert result.ok and result.stdout.strip() == "ok"
    missing = LocalCommandRunner().run(["definitely-not-a-real-binary-xyz"], timeout=5)
    assert missing.returncode == 127


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs a POSIX bash"
)
def test_launch_command_returns_while_the_job_keeps_running(tmp_path: Path) -> None:
    """Regression: backgrounding the whole cd/activate chain kept the SSH channel (here: the
    captured pipes) open under bash until the job ended, so launches timed out."""
    code_dir, run_dir = tmp_path / "code", tmp_path / "run"
    code_dir.mkdir()
    run_dir.mkdir()
    (run_dir / "launch.sh").write_text("exec sleep 30\n", encoding="utf-8")
    host = RemoteHost(
        name="local",
        host="localhost",
        workspace=str(tmp_path),
        code_dir=str(code_dir),
        environment={"activate": "true"},
    )
    command = SSHClient(host, DryRunCommandRunner()).in_dir(str(run_dir), LAUNCH_COMMAND)
    started = time.monotonic()
    completed = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, timeout=20, check=True
    )
    pid = int(completed.stdout.strip())
    try:
        assert time.monotonic() - started < 10
        os.kill(pid, 0)  # the job is still running after the "session" closed
    finally:
        os.kill(pid, signal.SIGTERM)
