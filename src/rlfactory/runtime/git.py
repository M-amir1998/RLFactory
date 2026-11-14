"""Detect the code version (package version + Git SHA) recorded in every lineage record."""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from rlfactory import __version__
from rlfactory.domain.common import CodeVersion

GIT_SHA_ENV = "RLFACTORY_GIT_SHA"


def _run_git(git: str, args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        [git, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return completed.stdout.strip()


@lru_cache(maxsize=8)
def detect_code_version(repo_dir: Path | None = None) -> CodeVersion:
    """Return the RLFactory version and, when available, the Git SHA and dirty flag.

    ``RLFACTORY_GIT_SHA`` takes precedence so container images and CI builds (which may not ship a
    ``.git`` directory) can still record exact lineage.
    """
    env_sha = os.environ.get(GIT_SHA_ENV)
    if env_sha:
        return CodeVersion(rlfactory_version=__version__, git_sha=env_sha.strip(), git_dirty=None)
    git = shutil.which("git")
    if git is None:
        return CodeVersion(rlfactory_version=__version__)
    cwd = repo_dir or Path(__file__).resolve().parent
    try:
        sha = _run_git(git, ["rev-parse", "HEAD"], cwd)
        status = _run_git(git, ["status", "--porcelain", "--untracked-files=no"], cwd)
    except (subprocess.SubprocessError, OSError):
        return CodeVersion(rlfactory_version=__version__)
    return CodeVersion(rlfactory_version=__version__, git_sha=sha or None, git_dirty=bool(status))
