"""Environment report for ``rlfactory doctor``. Imports nothing heavy (no torch import)."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import psutil

from rlfactory import __version__
from rlfactory.config.settings import Settings
from rlfactory.runtime.disk import GIB, free_bytes
from rlfactory.runtime.git import detect_code_version
from rlfactory.runtime.hf_cache import resolve_hf_cache_dir, scan_cache
from rlfactory.sandbox.docker_sandbox import DEFAULT_IMAGE, DockerSandbox
from rlfactory.telemetry.system import query_local_gpus

Status = Literal["ok", "info", "warn", "missing"]

OPTIONAL_PACKAGES = (
    ("torch", "training/inference (trl extra)"),
    ("transformers", "trl extra"),
    ("trl", "TRL training backend"),
    ("peft", "LoRA/QLoRA"),
    ("accelerate", "distributed launch"),
    ("datasets", "TRL data loading"),
    ("safetensors", "safe weight format"),
    ("bitsandbytes", "QLoRA (Linux + CUDA only)"),
    ("vllm", "rollout server (GPU host)"),
    ("sglang", "rollout server (GPU host)"),
    ("verl", "verl backend (separate GPU-host env)"),
    ("openrlhf", "OpenRLHF backend (separate GPU-host env)"),
)


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    status: Status
    detail: str


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _docker_checks() -> list[Check]:
    if shutil.which("docker") is None:
        return [
            Check(
                "sandbox", "docker", "info", "docker CLI not found; subprocess sandbox will be used"
            )
        ]
    sandbox = DockerSandbox()
    if not sandbox.available():
        return [Check("sandbox", "docker", "warn", "docker CLI present but daemon not reachable")]
    try:
        version = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        version = "unknown"
    image = sandbox.image_present()
    return [
        Check("sandbox", "docker", "ok", f"daemon {version}"),
        Check(
            "sandbox",
            "sandbox image",
            "ok" if image else "info",
            f"{DEFAULT_IMAGE} {'present' if image else f'not pulled (docker pull {DEFAULT_IMAGE})'}",
        ),
    ]


def run_checks(settings: Settings, *, include_docker: bool = True) -> list[Check]:
    checks: list[Check] = []
    python_ok = sys.version_info >= (3, 11)
    checks.append(
        Check("system", "python", "ok" if python_ok else "warn", platform.python_version())
    )
    checks.append(
        Check(
            "system",
            "platform",
            "info",
            f"{platform.system()} {platform.release()} {platform.machine()}",
        )
    )
    checks.append(Check("system", "cpu", "info", f"{os.cpu_count()} logical CPUs"))
    memory = psutil.virtual_memory()
    checks.append(
        Check(
            "system",
            "ram",
            "ok" if memory.total >= 8 * GIB else "warn",
            f"{memory.total / GIB:.1f} GiB total, {memory.available / GIB:.1f} GiB available",
        )
    )
    free = free_bytes(settings.results_dir)
    reserve = settings.disk_reserve_gb * GIB
    checks.append(
        Check(
            "system",
            "disk",
            "ok" if free > reserve else "warn",
            f"{free / GIB:.1f} GiB free at {settings.results_dir.resolve()} (reserve {settings.disk_reserve_gb} GiB)",
        )
    )
    gpus = query_local_gpus()
    checks.append(
        Check(
            "system",
            "gpu",
            "ok" if gpus else "info",
            ", ".join(f"{g.name} ({g.memory_total_gb:.0f} GB)" for g in gpus)
            or "no local GPU (normal: training runs on a remote GPU host)",
        )
    )
    code = detect_code_version()
    checks.append(
        Check(
            "rlfactory",
            "version",
            "ok",
            f"{__version__} git {code.git_sha[:12] if code.git_sha else 'unknown'}"
            + (" (dirty)" if code.git_dirty else ""),
        )
    )
    checks.append(
        Check(
            "rlfactory",
            "config dir",
            "ok" if settings.config_dir.is_dir() else "warn",
            str(settings.config_dir.resolve()),
        )
    )
    cache_dir = resolve_hf_cache_dir()
    cached = scan_cache(cache_dir)
    models = [repo for repo in cached if repo.repo_type == "model"]
    checks.append(
        Check(
            "huggingface",
            "cache",
            "info",
            f"{cache_dir} ({len(models)} models, {sum(r.size_bytes for r in models) / GIB:.2f} GiB)",
        )
    )
    if include_docker:
        checks.extend(_docker_checks())
    for package, purpose in OPTIONAL_PACKAGES:
        version = _version(package)
        checks.append(
            Check(
                "optional",
                package,
                "ok" if version else "missing",
                f"{version or 'not installed'} - {purpose}",
            )
        )
    return checks


def summarize(checks: list[Check]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return counts


def results_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True
