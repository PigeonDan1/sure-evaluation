"""TSE task-level evaluation routes.

The package keeps heavy compatibility metrics lazy so metric-only evaluation
images can import lightweight route modules without heavy dependencies.
"""

from __future__ import annotations

from typing import Any

from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples
from sure_eval.evaluation.tasks.tse.types import TSESample

__all__ = [
    "SISDRMetric",
    "TSESample",
    "evaluate_tse_samples",
]


def __getattr__(name: str) -> Any:
    if name == "SISDRMetric":
        from sure_eval.evaluation.tasks.tse import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")