"""Hugging Face cache discovery and a guard against accidental large checkpoint downloads."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from rlfactory.domain.model import ModelReference
from rlfactory.runtime.disk import GIB, DiskGuard


def resolve_hf_cache_dir(environ: dict[str, str] | None = None) -> Path:
    """Resolve the hub cache directory with the same precedence as ``huggingface_hub``."""
    env = dict(os.environ) if environ is None else environ
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class CachedRepo:
    repo_id: str
    repo_type: str
    path: Path
    size_bytes: int
    revisions: tuple[str, ...]


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            if not file_path.is_symlink():
                total += file_path.stat().st_size
    return total


def scan_cache(cache_dir: Path | None = None) -> list[CachedRepo]:
    """List cached repos. Sizes count ``blobs/`` only, so symlinked snapshots are not doubled."""
    root = cache_dir or resolve_hf_cache_dir()
    if not root.is_dir():
        return []
    repos: list[CachedRepo] = []
    for entry in sorted(root.iterdir()):
        prefix, sep, rest = entry.name.partition("--")
        if not sep or prefix not in {"models", "datasets", "spaces"} or not entry.is_dir():
            continue
        snapshots = entry / "snapshots"
        revisions = (
            tuple(sorted(child.name for child in snapshots.iterdir())) if snapshots.is_dir() else ()
        )
        blobs = entry / "blobs"
        size = _dir_size(blobs) if blobs.is_dir() else _dir_size(entry)
        repos.append(
            CachedRepo(
                repo_id=rest.replace("--", "/"),
                repo_type=prefix.rstrip("s"),
                path=entry,
                size_bytes=size,
                revisions=revisions,
            )
        )
    return repos


def find_cached_model(model_id: str, cache_dir: Path | None = None) -> CachedRepo | None:
    for repo in scan_cache(cache_dir):
        if repo.repo_type == "model" and repo.repo_id == model_id:
            return repo
    return None


@dataclass(frozen=True)
class DownloadPlan:
    model: ModelReference
    cache_dir: Path
    already_cached: bool
    expected_size_bytes: int | None
    free_bytes: int
    allowed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        size = (
            f"{self.expected_size_bytes / GIB:.2f} GiB"
            if self.expected_size_bytes is not None
            else "unknown"
        )
        lines = [
            f"model:          {self.model.describe()}",
            f"expected size:  {size}",
            f"free disk:      {self.free_bytes / GIB:.2f} GiB",
            f"cache path:     {self.cache_dir}",
            f"already cached: {self.already_cached}",
            f"allowed:        {self.allowed}",
        ]
        lines.extend(f"  - {reason}" for reason in self.reasons)
        return "\n".join(lines)


class CheckpointGuard:
    """Decide whether a checkpoint download may proceed. Never downloads anything itself."""

    def __init__(self, disk_guard: DiskGuard, cache_dir: Path | None = None) -> None:
        self.disk_guard = disk_guard
        self.cache_dir = cache_dir or resolve_hf_cache_dir()

    def plan(
        self,
        models: list[ModelReference],
        expected_sizes: dict[str, int] | None = None,
        *,
        allow_download: bool = False,
    ) -> list[DownloadPlan]:
        sizes = expected_sizes or {}
        to_download = [m for m in models if find_cached_model(m.name, self.cache_dir) is None]
        plans: list[DownloadPlan] = []
        for model in models:
            cached = find_cached_model(model.name, self.cache_dir) is not None
            expected = sizes.get(model.name)
            check = self.disk_guard.check(self.cache_dir, expected or 0)
            reasons: list[str] = []
            if not cached:
                if not allow_download:
                    reasons.append("download requires explicit --allow-download")
                if len(to_download) > 1:
                    reasons.append("refusing to download more than one uncached checkpoint at once")
                if expected is None:
                    reasons.append("expected size unknown; verify on the model card first")
                if not check.ok:
                    reasons.append(f"insufficient disk: {check.describe()}")
            if model.trust_remote_code:
                reasons.append("trust_remote_code=True requested: review the repository code first")
            allowed = cached or (
                allow_download and len(to_download) <= 1 and check.ok and expected is not None
            )
            plans.append(
                DownloadPlan(
                    model=model,
                    cache_dir=self.cache_dir,
                    already_cached=cached,
                    expected_size_bytes=expected,
                    free_bytes=check.free_bytes,
                    allowed=allowed,
                    reasons=tuple(reasons),
                )
            )
        return plans
