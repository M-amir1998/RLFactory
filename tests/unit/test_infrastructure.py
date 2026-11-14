"""Telemetry, runtime (disk, HF cache, code version, doctor), settings, config loading, reporting."""

from __future__ import annotations

import io
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pydantic import BaseModel

from rlfactory.config.loader import dump_yaml, load_model, load_yaml, read_json_model, write_json
from rlfactory.config.settings import Settings
from rlfactory.domain.model import ModelReference
from rlfactory.errors import ConfigError, InsufficientDiskError
from rlfactory.reporting.plots import (
    Series,
    grouped_bar_chart,
    histogram_chart,
    horizontal_bar_chart,
    line_chart,
)
from rlfactory.reporting.report import generate_report
from rlfactory.runtime.disk import DiskGuard, estimate_dataset_bytes, estimate_rollout_bytes
from rlfactory.runtime.doctor import run_checks, summarize
from rlfactory.runtime.git import GIT_SHA_ENV, detect_code_version
from rlfactory.runtime.hf_cache import (
    CheckpointGuard,
    find_cached_model,
    resolve_hf_cache_dir,
    scan_cache,
)
from rlfactory.telemetry.logging import (
    REDACTED,
    configure_logging,
    get_logger,
    log_context,
    log_event,
    redact,
)
from rlfactory.telemetry.metrics import MetricsRegistry, RolloutMetrics
from rlfactory.telemetry.system import ResourceMonitor, collect_hardware_info, parse_nvidia_smi

# Deliberately fake, non-functional fixtures for the redaction filter. They are assembled at
# runtime so that no secret-shaped literal exists anywhere in the repository.
_ALPHA = "abcdefghijklmnopqrstuvwxyz"
FAKE_SECRETS = [
    "hf_" + _ALPHA + "0123",
    "sk-" + _ALPHA.upper()[:24],
    "AKIA" + _ALPHA.upper()[:16],
    "ghp_" + _ALPHA,
    "Bearer " + _ALPHA,
    "-----BEGIN RSA " + "PRIVATE KEY-----\nabc\n-----END RSA " + "PRIVATE KEY-----",
]


@pytest.mark.parametrize("secret", FAKE_SECRETS)
def test_redaction(secret: str) -> None:
    assert secret not in redact(f"value={secret} end") and REDACTED in redact(secret)


def test_text_artefacts_use_lf_line_endings(tmp_path: Path) -> None:
    # Byte-identical artefacts across platforms: no CRLF translation on Windows.
    write_json({"values": [1, 2], "name": "x"}, tmp_path / "out.json")
    dump_yaml({"values": [1, 2], "name": "x"}, tmp_path / "out.yaml")
    for name in ("out.json", "out.yaml"):
        data = (tmp_path / name).read_bytes()
        assert b"\n" in data and b"\r" not in data


def test_structured_json_logging_with_context() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "json", stream)
    logger = get_logger("test")
    with log_context(run_id="run-1", stage="rollouts"):
        log_event(logger, "trajectory done", trajectory_id="t1", reward=1.0, token="hf_" + "a" * 25)
    record = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert (
        record["run_id"] == "run-1"
        and record["stage"] == "rollouts"
        and record["trajectory_id"] == "t1"
    )
    assert record["token"] == REDACTED and record["level"] == "INFO"
    text_stream = io.StringIO()
    configure_logging("INFO", "text", text_stream)
    with log_context(stage="x"):
        get_logger("rlfactory.t").info("hello %s", "sk-" + "b" * 25)
    assert "stage=x" in text_stream.getvalue() and REDACTED in text_stream.getvalue()
    logging.getLogger("rlfactory").handlers.clear()


def test_metrics_registry_and_prometheus_text() -> None:
    metrics = RolloutMetrics()
    metrics.trajectories.inc(status="completed", domain="python")
    metrics.policy_latency.observe(0.2)
    metrics.policy_latency.observe(1.5)
    metrics.reward.observe(1.0)
    text = metrics.registry.to_prometheus()
    assert 'rlfactory_trajectories_total{domain="python",status="completed"} 1.0' in text
    assert 'rlfactory_policy_latency_seconds_bucket{le="+Inf"} 2' in text
    assert metrics.policy_latency.quantile(0.5) == 0.2 and metrics.policy_latency.count() == 2
    registry = MetricsRegistry()
    registry.gauge("g", "help").set(3.0, host="a")
    assert registry.gauge("g", "help").value(host="a") == 3.0
    with pytest.raises(TypeError):
        registry.counter("g", "help")
    with pytest.raises(ValueError, match="only increase"):
        registry.counter("c", "h").inc(-1)


def test_system_info_and_resource_monitor(tmp_path: Path) -> None:
    info = collect_hardware_info()
    assert info.cpu_count >= 1 and info.ram_gb > 0
    assert parse_nvidia_smi("NVIDIA A100, 81920, 1024\ngarbage\n")[0].memory_total_gb == 80.0
    with ResourceMonitor(tmp_path, interval_s=0.01) as monitor:
        sum(range(10000))
    assert monitor.usage is not None and monitor.usage.peak_rss_gb > 0
    assert set(monitor.usage.as_dict()) >= {"wall_time_s", "peak_rss_gb"}


def test_disk_guard(tmp_path: Path) -> None:
    assert DiskGuard(0).ensure(tmp_path / "new" / "dir", 1024, "test").ok
    with pytest.raises(InsufficientDiskError, match="refusing"):
        DiskGuard(10**9).ensure(tmp_path, 0, "huge reserve")
    assert estimate_rollout_bytes(10) == 80_000 and estimate_dataset_bytes(-1) == 0
    with pytest.raises(ValueError, match="non-negative"):
        DiskGuard(-1)


def _fake_cache(root: Path) -> None:
    blobs = root / "models--org--tiny" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc").write_bytes(b"0" * 2048)
    (root / "models--org--tiny" / "snapshots" / "rev1").mkdir(parents=True)
    (root / "datasets--org--data").mkdir()
    (root / "not-a-repo").mkdir()


def test_hf_cache_scan_and_checkpoint_guard(tmp_path: Path) -> None:
    _fake_cache(tmp_path)
    repos = {r.repo_id: r for r in scan_cache(tmp_path)}
    assert repos["org/tiny"].size_bytes == 2048 and repos["org/tiny"].revisions == ("rev1",)
    assert repos["org/data"].repo_type == "dataset"
    assert (
        find_cached_model("org/tiny", tmp_path) is not None
        and find_cached_model("x/y", tmp_path) is None
    )
    guard = CheckpointGuard(DiskGuard(0), tmp_path)
    (cached,) = guard.plan([ModelReference(name="org/tiny")])
    assert cached.allowed and cached.already_cached
    (new,) = guard.plan([ModelReference(name="org/new")])
    assert not new.allowed and any("--allow-download" in r for r in new.reasons)
    (sized,) = guard.plan([ModelReference(name="org/new")], {"org/new": 1000}, allow_download=True)
    assert sized.allowed
    two = guard.plan(
        [ModelReference(name="a/b"), ModelReference(name="c/d")],
        {"a/b": 1, "c/d": 1},
        allow_download=True,
    )
    assert not any(p.allowed for p in two)
    (remote_code,) = guard.plan([ModelReference(name="org/tiny", trust_remote_code=True)])
    assert any("trust_remote_code" in r for r in remote_code.reasons)
    assert "model:" in cached.describe()
    assert resolve_hf_cache_dir({"HF_HUB_CACHE": str(tmp_path)}) == tmp_path
    assert resolve_hf_cache_dir({"HF_HOME": str(tmp_path)}) == tmp_path / "hub"


def test_code_version_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    detect_code_version.cache_clear()
    monkeypatch.setenv(GIT_SHA_ENV, "deadbeef")
    assert detect_code_version().git_sha == "deadbeef"
    detect_code_version.cache_clear()


def test_settings_and_run_dir_validation() -> None:
    settings = Settings.from_env(
        {
            "RLFACTORY_RESULTS_DIR": "out",
            "RLFACTORY_DISK_RESERVE_GB": "1.5",
            "RLFACTORY_LOG_FORMAT": "json",
        }
    )
    assert settings.results_dir == Path("out") and settings.disk_reserve_gb == 1.5
    assert settings.run_dir("abc") == Path("out") / "abc"
    for bad in ("", "../x", "a/b", "a\\b"):
        with pytest.raises(ValueError, match="invalid run id"):
            settings.run_dir(bad)


class _Model(BaseModel):
    value: int


def test_config_loader_errors_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("value: 3\n", encoding="utf-8")
    assert load_model(path, _Model).value == 3
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(path)
    path.write_text("value: [\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml(path)
    path.write_text("value: nope\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid _Model"):
        load_model(path, _Model)
    path.write_text("!!python/object/apply:os.system ['echo hacked']\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml(path)
    with pytest.raises(ConfigError, match="cannot read"):
        load_yaml(tmp_path / "missing.yaml")
    dump_yaml(_Model(value=4), tmp_path / "o.yaml")
    assert load_model(tmp_path / "o.yaml", _Model).value == 4
    write_json(_Model(value=5), tmp_path / "o.json")
    assert read_json_model(tmp_path / "o.json", _Model).value == 5


def test_doctor_runs_without_docker() -> None:
    checks = run_checks(Settings(), include_docker=False)
    names = {c.name for c in checks}
    assert {"python", "ram", "disk", "gpu", "trl", "vllm"} <= names
    assert sum(summarize(checks).values()) == len(checks)


def test_svg_charts_are_well_formed() -> None:
    charts = [
        grouped_bar_chart(
            "t", ["a", "b"], [Series("base", [0.5, None]), Series("cand", [0.7, 0.1])]
        ),
        histogram_chart("h", [(0.0, 0.5, 3), (0.5, 1.0, 0)], x_label="reward"),
        horizontal_bar_chart("f", [("x", 3), ("y", 0)]),
        line_chart("l", ["0", "1", "2"], [0.4, None, 0.6]),
    ]
    for svg in charts:
        root = ET.fromstring(svg)  # noqa: S314 - parsing our own generated SVG
        assert root.tag.endswith("svg") and "prefers-color-scheme: dark" in svg and "<title>" in svg
    with pytest.raises(ValueError, match="one or two series"):
        grouped_bar_chart("t", ["a"], [Series(str(i), [0.1]) for i in range(3)])


def test_report_renders_from_partial_artifacts(tmp_path: Path) -> None:
    path = generate_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "SIMULATED RESULTS" in text and "## Limitations" in text and "Not run." in text
