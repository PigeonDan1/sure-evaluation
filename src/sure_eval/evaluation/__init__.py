"""Evaluation module for SURE-EVAL."""

__all__ = [
    "SUREEvaluator",
    "MetricRegistry",
    "MetricResult",
    "RPSManager",
    "RPSCalculator",
]


def __getattr__(name: str):
    """Load evaluation entrypoints lazily so metric-only imports stay light."""
    if name == "SUREEvaluator":
        from sure_eval.evaluation.sure_evaluator import SUREEvaluator

        return SUREEvaluator
    if name == "MetricRegistry":
        from sure_eval.evaluation.registry import MetricRegistry

        return MetricRegistry
    if name == "MetricResult":
        from sure_eval.evaluation.base import MetricResult

        return MetricResult
    if name in {"RPSManager", "RPSCalculator"}:
        from sure_eval.evaluation.rps import RPSCalculator, RPSManager

        return {"RPSManager": RPSManager, "RPSCalculator": RPSCalculator}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
