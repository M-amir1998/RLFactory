"""Host hardware description and lightweight resource sampling (CPU, RAM, disk, GPU)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import psutil

from rlfactory.domain.experiment import HardwareInfo

GIB = 1024**3


@dataclass(frozen=True)
class GPUInfo:
    name: str
    memory_total_gb: float
    memory_used_gb: float


def parse_nvidia_smi(output: str) -> list[GPUInfo]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits``."""
    gpus: list[GPUInfo] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            total_mib, used_mib = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        gpus.append(GPUInfo(parts[0], total_mib / 1024, used_mib / 1024))
    return gpus


def query_local_gpus(timeout_s: float = 5.0) -> list[GPUInfo]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        completed = subprocess.run(
            [smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    return parse_nvidia_smi(completed.stdout)


def collect_hardware_info() -> HardwareInfo:
    gpus = query_local_gpus()
    return HardwareInfo(
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=round(psutil.virtual_memory().total / GIB, 2),
        gpu=", ".join(g.name for g in gpus) or None,
        gpu_memory_gb=round(sum(g.memory_total_gb for g in gpus), 2) if gpus else None,
    )


@dataclass(frozen=True)
class ResourceUsage:
    wall_time_s: float
    peak_rss_gb: float
    mean_cpu_percent: float
    peak_system_ram_percent: float
    disk_free_gb_start: float
    disk_free_gb_end: float

    def as_dict(self) -> dict[str, float]:
        return {
            "wall_time_s": round(self.wall_time_s, 3),
            "peak_rss_gb": round(self.peak_rss_gb, 3),
            "mean_cpu_percent": round(self.mean_cpu_percent, 1),
            "peak_system_ram_percent": round(self.peak_system_ram_percent, 1),
            "disk_free_gb_start": round(self.disk_free_gb_start, 2),
            "disk_free_gb_end": round(self.disk_free_gb_end, 2),
        }


class ResourceMonitor:
    """Sample this process's RSS, CPU and system RAM in a daemon thread while a stage runs."""

    def __init__(self, disk_path: Path, interval_s: float = 0.5) -> None:
        self.disk_path = disk_path
        self.interval_s = interval_s
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0
        self._peak_ram_percent = 0.0
        self._cpu_samples: list[float] = []
        self._start = 0.0
        self._disk_start = 0.0
        self.usage: ResourceUsage | None = None

    def _disk_free_gb(self) -> float:
        path = self.disk_path if self.disk_path.exists() else Path.cwd()
        return shutil.disk_usage(path).free / GIB

    def _sample(self) -> None:
        try:
            self._peak_rss = max(self._peak_rss, self._process.memory_info().rss)
            self._peak_ram_percent = max(self._peak_ram_percent, psutil.virtual_memory().percent)
            self._cpu_samples.append(self._process.cpu_percent(interval=None))
        except psutil.Error:  # pragma: no cover - process introspection is best-effort
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> ResourceMonitor:
        self._start = time.perf_counter()
        self._disk_start = self._disk_free_gb()
        self._process.cpu_percent(interval=None)
        self._sample()
        self._thread = threading.Thread(target=self._run, name="rlfactory-resources", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()
        cpu = [c for c in self._cpu_samples if c >= 0]
        self.usage = ResourceUsage(
            wall_time_s=time.perf_counter() - self._start,
            peak_rss_gb=self._peak_rss / GIB,
            mean_cpu_percent=sum(cpu) / len(cpu) if cpu else 0.0,
            peak_system_ram_percent=self._peak_ram_percent,
            disk_free_gb_start=self._disk_start,
            disk_free_gb_end=self._disk_free_gb(),
        )
