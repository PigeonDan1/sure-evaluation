# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Request-scoped cumulative counters and per-line immutable diagnostics."""

from dataclasses import dataclass


COUNTER_NAMES = (
    "processed_lines",
    "map_rule_hits",
    "number_matches",
    "number_cache_hits",
    "number_cache_misses",
    "fallbacks",
)


@dataclass
class RuntimeCounters:
    """Belongs to one compiled pipeline and is not shared across requests."""

    processed_lines: int = 0
    map_rule_hits: int = 0
    number_matches: int = 0
    number_cache_hits: int = 0
    number_cache_misses: int = 0
    fallbacks: int = 0

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in COUNTER_NAMES:
            raise ValueError(f"unknown normalization counter: {name}")
        setattr(self, name, getattr(self, name) + amount)

    def snapshot(self) -> tuple[int, ...]:
        return tuple(getattr(self, name) for name in COUNTER_NAMES)


@dataclass(frozen=True)
class NormalizationDiagnostics:
    """Per-line counters; not exposed in default CLI output."""

    engine: str
    removed_line: bool
    processed_lines: int = 0
    map_rule_hits: int = 0
    number_matches: int = 0
    number_cache_hits: int = 0
    number_cache_misses: int = 0
    fallbacks: int = 0


@dataclass(frozen=True)
class PipelineMetrics:
    """Cumulative observable metrics since a compiled pipeline was created."""

    engine: str
    processed_lines: int
    removed_lines: int
    map_rule_hits: int
    number_matches: int
    number_cache_hits: int
    number_cache_misses: int
    fallbacks: int
    number_cache_entries: int
    number_cache_capacity: int
