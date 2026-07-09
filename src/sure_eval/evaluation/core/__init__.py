"""Core building blocks for versioned evaluation pipelines."""

from __future__ import annotations

__all__ = [
    "EvaluationFiles",
    "EvaluationReport",
    "KeyTextFiles",
    "MetricInputContract",
    "PipelineNodeResult",
    "PipelineSpec",
    "run_pipeline",
]


def __getattr__(name: str):
    if name in {
        "EvaluationFiles",
        "EvaluationReport",
        "KeyTextFiles",
        "MetricInputContract",
        "PipelineNodeResult",
        "PipelineSpec",
    }:
        from sure_eval.evaluation.core.types import (
            EvaluationFiles,
            EvaluationReport,
            KeyTextFiles,
            MetricInputContract,
            PipelineNodeResult,
            PipelineSpec,
        )

        return {
            "EvaluationFiles": EvaluationFiles,
            "EvaluationReport": EvaluationReport,
            "KeyTextFiles": KeyTextFiles,
            "MetricInputContract": MetricInputContract,
            "PipelineNodeResult": PipelineNodeResult,
            "PipelineSpec": PipelineSpec,
        }[name]
    if name == "run_pipeline":
        from sure_eval.evaluation.core.pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(name)
