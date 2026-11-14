"""Disk-safety checks: estimate output sizes and enforce a configurable free-space reserve."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rlfactory.errors import InsufficientDiskError

GIB = 1024**3

# Conservative per-record sizes (bytes, zstd-compressed Parquet / JSONL), measured on the CPU
# end-to-end pipeline and rounded up. Override per call when better estimates exist.
BYTES_PER_TRAJECTORY = 8_000
BYTES_PER_DATASET_RECORD = 4_000


@dataclass(frozen=True)
class DiskCheck:
    path: Path
    free_bytes: int
    required_bytes: int
    reserve_bytes: int

    @property
    def ok(self) -> bool:
        return self.free_bytes - self.required_bytes >= self.reserve_bytes

    def describe(self) -> str:
        return (
            f"free={self.free_bytes / GIB:.2f} GiB, required~{self.required_bytes / GIB:.3f} GiB, "
            f"reserve={self.reserve_bytes / GIB:.2f} GiB at {self.path}"
        )


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return candidate


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(_existing_ancestor(path)).free


class DiskGuard:
    """Refuse operations that would push free space below ``reserve_gb``."""

    def __init__(self, reserve_gb: float = 5.0) -> None:
        if reserve_gb < 0:
            raise ValueError("reserve_gb must be non-negative")
        self.reserve_bytes = int(reserve_gb * GIB)

    def check(self, path: Path, required_bytes: int) -> DiskCheck:
        return DiskCheck(
            path=path,
            free_bytes=free_bytes(path),
            required_bytes=max(0, required_bytes),
            reserve_bytes=self.reserve_bytes,
        )

    def ensure(self, path: Path, required_bytes: int, operation: str) -> DiskCheck:
        result = self.check(path, required_bytes)
        if not result.ok:
            raise InsufficientDiskError(f"refusing {operation}: {result.describe()}")
        return result


def estimate_rollout_bytes(
    num_trajectories: int, bytes_per_trajectory: int = BYTES_PER_TRAJECTORY
) -> int:
    return max(0, num_trajectories) * bytes_per_trajectory


def estimate_dataset_bytes(
    num_records: int, bytes_per_record: int = BYTES_PER_DATASET_RECORD
) -> int:
    return max(0, num_records) * bytes_per_record
