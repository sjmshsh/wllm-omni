from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator


@dataclass(slots=True)
class ProfileMetric:
    count: int = 0
    total_s: float = 0.0
    min_s: float | None = None
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.min_s = elapsed_s if self.min_s is None else min(self.min_s, elapsed_s)
        self.max_s = max(self.max_s, elapsed_s)

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0


class RequestProfiler:
    """Small per-request profiler for runner/executor development."""

    def __init__(self, req_id: str):
        self.req_id = req_id
        self.metrics: OrderedDict[str, ProfileMetric] = OrderedDict()
        self.metadata: dict[str, object] = {}

    @contextmanager
    def stage(self, name: str, sync_fn: Callable[[], None] | None = None) -> Iterator[None]:
        if sync_fn is not None:
            sync_fn()
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync_fn is not None:
                sync_fn()
            self.record(name, time.perf_counter() - start)

    def record(self, name: str, elapsed_s: float) -> None:
        metric = self.metrics.get(name)
        if metric is None:
            metric = ProfileMetric()
            self.metrics[name] = metric
        metric.add(elapsed_s)

    def set_metadata(self, **kwargs: object) -> None:
        self.metadata.update(kwargs)

    def summary_lines(self) -> list[str]:
        lines = [f"[wllm-omni][profile] req_id={self.req_id}"]
        if self.metadata:
            metadata = " ".join(f"{key}={value}" for key, value in self.metadata.items())
            lines.append(f"[wllm-omni][profile] metadata {metadata}")
        lines.append(
            "[wllm-omni][profile] "
            f"{'stage':<34} {'count':>5} {'total_s':>10} {'mean_ms':>10} {'min_ms':>10} {'max_ms':>10}"
        )
        for name, metric in self.metrics.items():
            min_s = metric.min_s or 0.0
            lines.append(
                "[wllm-omni][profile] "
                f"{name:<34} {metric.count:>5} {metric.total_s:>10.4f} "
                f"{metric.mean_s * 1000:>10.2f} {min_s * 1000:>10.2f} {metric.max_s * 1000:>10.2f}"
            )
        return lines
