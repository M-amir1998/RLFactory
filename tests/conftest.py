"""Test configuration: directory-based markers, offline defaults, shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Never touch the network or download models from tests.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_DIRECTORY_MARKERS = {
    "unit": "unit",
    "property": "property",
    "integration": "integration",
    "e2e": "e2e",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        parts = Path(str(item.fspath)).parts
        for directory, marker in _DIRECTORY_MARKERS.items():
            if directory in parts:
                item.add_marker(getattr(pytest.mark, marker))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config_dir() -> Path:
    return REPO_ROOT / "configs"
