"""Minimal thread-safe metrics registry with Prometheus text exposition.

Deliberately dependency-free: rollouts run on laptops and GPU hosts alike, and the exposition
format is simple enough that ``prometheus_client`` is not required. Metric files can be scraped
with node-exporter's textfile collector or simply archived with the run.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TypeVar

LabelKey = tuple[tuple[str, str], ...]

DEFAULT_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
REWARD_BUCKETS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5)


def _key(labels: dict[str, str]) -> LabelKey:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def _fmt_labels(key: LabelKey, extra: Iterable[tuple[str, str]] = ()) -> str:
    items = [*key, *extra]
    if not items:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in items)
    return "{" + inner + "}"


@dataclass
class Counter:
    name: str
    help: str
    _values: dict[LabelKey, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counters only increase")
        with self._lock:
            key = _key(labels)
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        lines += [f"{self.name}{_fmt_labels(k)} {v}" for k, v in sorted(self._values.items())]
        return lines


@dataclass
class Gauge:
    name: str
    help: str
    _values: dict[LabelKey, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float, **labels: str) -> None:
        with self._lock:
            self._values[_key(labels)] = value

    def value(self, **labels: str) -> float | None:
        return self._values.get(_key(labels))

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        lines += [f"{self.name}{_fmt_labels(k)} {v}" for k, v in sorted(self._values.items())]
        return lines


@dataclass
class _HistogramState:
    bucket_counts: list[int]
    count: int = 0
    total: float = 0.0
    values: list[float] = field(default_factory=list)


@dataclass
class Histogram:
    name: str
    help: str
    buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS
    keep_values: int = 10_000
    _states: dict[LabelKey, _HistogramState] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, **labels: str) -> None:
        if math.isnan(value):
            return
        with self._lock:
            key = _key(labels)
            state = self._states.setdefault(key, _HistogramState([0] * len(self.buckets)))
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    state.bucket_counts[i] += 1
            state.count += 1
            state.total += value
            if len(state.values) < self.keep_values:
                state.values.append(value)

    def count(self, **labels: str) -> int:
        state = self._states.get(_key(labels))
        return state.count if state else 0

    def quantile(self, q: float, **labels: str) -> float | None:
        state = self._states.get(_key(labels))
        if not state or not state.values:
            return None
        ordered = sorted(state.values)
        index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return ordered[index]

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, state in sorted(self._states.items()):
            for bound, bucket_count in zip(self.buckets, state.bucket_counts, strict=True):
                lines.append(
                    f"{self.name}_bucket{_fmt_labels(key, [('le', str(bound))])} {bucket_count}"
                )
            lines.append(f"{self.name}_bucket{_fmt_labels(key, [('le', '+Inf')])} {state.count}")
            lines.append(f"{self.name}_sum{_fmt_labels(key)} {state.total}")
            lines.append(f"{self.name}_count{_fmt_labels(key)} {state.count}")
        return lines


MetricT = TypeVar("MetricT", Counter, Gauge, Histogram)


class MetricsRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help_text: str) -> Counter:
        return self._get_or_create(name, lambda: Counter(name, help_text), Counter)

    def gauge(self, name: str, help_text: str) -> Gauge:
        return self._get_or_create(name, lambda: Gauge(name, help_text), Gauge)

    def histogram(
        self, name: str, help_text: str, buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS
    ) -> Histogram:
        return self._get_or_create(name, lambda: Histogram(name, help_text, buckets), Histogram)

    def _get_or_create(
        self, name: str, factory: Callable[[], MetricT], kind: type[MetricT]
    ) -> MetricT:
        with self._lock:
            existing = self._metrics.get(name)
            if existing is None:
                created = factory()
                self._metrics[name] = created
                return created
            if not isinstance(existing, kind):
                raise TypeError(f"metric {name} already registered as {type(existing).__name__}")
            return existing

    def to_prometheus(self) -> str:
        lines: list[str] = []
        for name in sorted(self._metrics):
            lines.extend(self._metrics[name].render())
        return "\n".join(lines) + "\n"


class RolloutMetrics:
    """Named rollout metrics shared by the orchestrator, worker and reporting."""

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        self.registry = registry or MetricsRegistry()
        r = self.registry
        self.trajectories = r.counter(
            "rlfactory_trajectories_total", "Trajectories finished, by status and domain"
        )
        self.failures = r.counter(
            "rlfactory_failures_total", "Failures by kind (model vs infrastructure)"
        )
        self.retries = r.counter("rlfactory_retries_total", "Retried rollout attempts by kind")
        self.policy_latency = r.histogram(
            "rlfactory_policy_latency_seconds", "Policy generation latency"
        )
        self.environment_latency = r.histogram(
            "rlfactory_environment_latency_seconds", "Environment step latency (incl. sandbox)"
        )
        self.verifier_latency = r.histogram(
            "rlfactory_verifier_latency_seconds", "Verifier latency"
        )
        self.reward = r.histogram("rlfactory_reward", "Scalar reward distribution", REWARD_BUCKETS)
        self.sandbox_failures = r.counter(
            "rlfactory_sandbox_failures_total", "Sandbox infrastructure failures"
        )
