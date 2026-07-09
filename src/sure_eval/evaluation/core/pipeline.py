"""Sequential runner for versioned evaluation pipelines."""

from __future__ import annotations

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult, PipelineSpec


def run_pipeline(spec: PipelineSpec, files: KeyTextFiles) -> tuple[KeyTextFiles, tuple[PipelineNodeResult, ...]]:
    """Run a pipeline and collect node trace entries."""

    current = files
    trace: list[PipelineNodeResult] = []
    for node in spec.nodes:
        current, result = node(current)
        trace.append(result)
    return current, tuple(trace)
