from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple


class _KernelProfiler:
    """Tracks backend kernel launches."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self._counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def reset(self) -> None:
        self._counts.clear()

    def record(self, device_name: str, op_name: str) -> None:
        if self.enabled:
            self._counts[(device_name, op_name)] += 1

    def counts(self) -> Dict[Tuple[str, str], int]:
        return dict(self._counts)

    def total(self, device_name: str | None = None) -> int:
        if device_name is None:
            return sum(self._counts.values())
        return sum(
            count for (dev, _), count in self._counts.items() if dev == device_name
        )


KERNEL_PROFILER = _KernelProfiler()


def enable_kernel_profiler() -> None:
    KERNEL_PROFILER.enable()


def disable_kernel_profiler() -> None:
    KERNEL_PROFILER.disable()


def reset_kernel_profiler() -> None:
    KERNEL_PROFILER.reset()


def get_kernel_counts() -> Dict[Tuple[str, str], int]:
    return KERNEL_PROFILER.counts()


def get_total_kernel_count(device_name: str | None = None) -> int:
    return KERNEL_PROFILER.total(device_name=device_name)
