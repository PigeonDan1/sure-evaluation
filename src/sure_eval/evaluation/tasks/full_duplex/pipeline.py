"""Compose reusable timeline normalization and FD-Bench timing nodes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sure_eval.evaluation.core.types import (
    EvaluationFiles,
    EvaluationReport,
    MetricInputContract,
    PipelineNodeResult,
)
from sure_eval.evaluation.nodes.normalization.duplex_timeline import normalize_duplex_timeline
from sure_eval.evaluation.nodes.normalization.duplex_timeline.measurement_protocol import (
    summarize_measurement_protocol,
)
from sure_eval.evaluation.nodes.normalization.duplex_timeline.types import json_safe
from sure_eval.evaluation.nodes.scoring.fdbench_timing import PROTOCOL, score_fdbench_timing
from sure_eval.evaluation.pipeline_identity import (
    build_atomic_pipeline_id,
    canonical_metric,
    component_trace_ids,
    node_component,
)

_PROTOCOL = PROTOCOL
_METRICS = {"ird", "fsed", "ert", "eit"}
_RESULT_SUMMARY_FIELDS = (
    "score", "unit", "sample_count", "positive_tick_count", "zero_ms_count", "status", "reason",
)
_CONTRACT = MetricInputContract(
    metric_id="task/full_duplex_jsonl",
    required_roles=("reference_jsonl", "sample_output"),
    row_format="duplex_timeline_jsonl",
    alignment_key="key",
    aggregation="pooled_event_median_ms",
    purpose="full_duplex_conversation_timing",
)


def evaluate_full_duplex_files(
    *,
    reference_jsonl: str | Path | None = None,
    sample_output: str | Path | None = None,
    metric: str = "ird",
    protocol: str = _PROTOCOL,
) -> EvaluationReport:
    """Score saved user/system timelines without calling a speech model."""

    normalized_metric = canonical_metric(metric)
    if normalized_metric not in _METRICS:
        raise ValueError(f"Unsupported FULL_DUPLEX metric: {metric}")
    if protocol != _PROTOCOL:
        raise ValueError(f"Unsupported FULL_DUPLEX protocol: {protocol}")
    roles = {}
    if reference_jsonl is not None:
        roles["reference_jsonl"] = str(reference_jsonl)
    if sample_output is not None:
        roles["sample_output"] = str(sample_output)
    input_files = EvaluationFiles(roles=roles)
    _CONTRACT.validate(input_files)

    bundle, normalization_result = normalize_duplex_timeline(
        input_files.roles["reference_jsonl"], input_files.roles["sample_output"]
    )
    scoring_result = score_fdbench_timing(bundle, metric=normalized_metric, protocol=protocol)
    scoring_details = scoring_result.details
    metadata_by_key = {
        conversation.key: json_safe({
            role: {name: value for name, value in metadata.items() if name != "measurement_protocol"}
            for role, metadata in conversation.metadata.items()
        })
        for conversation in bundle.conversations
    }
    # The standalone scorer retains its complete diagnostics. The task stores
    # those large records once, alongside provenance, and uses a compact trace.
    per_conversation = [
        {**row, "metadata": metadata_by_key[row["key"]]}
        for row in scoring_details["per_conversation"]
    ]
    components = _identity_components()
    return EvaluationReport(
        task="FULL_DUPLEX",
        language="n/a",
        metric=normalized_metric,
        score=float(scoring_result.details["score"]),
        pipeline_id=build_atomic_pipeline_id("full_duplex", "any", normalized_metric, components),
        pipeline_trace=(normalization_result, _scoring_summary(scoring_result)),
        input_contract=_CONTRACT,
        input_files=input_files,
        computation_node_ids=component_trace_ids(components),
        details={
            "results": scoring_details["results"],
            "evaluation_config": scoring_details["evaluation_config"],
            "input_sha256": {role: _sha256_file(path) for role, path in input_files.roles.items()},
            "per_conversation": per_conversation,
            "events": scoring_details["events"],
            "measurement_protocol": summarize_measurement_protocol(bundle),
            "input_contract": _CONTRACT.as_dict(),
            "input_files": input_files.as_dict(),
        },
    )


def _scoring_summary(result: PipelineNodeResult) -> PipelineNodeResult:
    details = result.details
    summary = {
        field: details[field]
        for field in (
            "score", "metric", "unit", "protocol", "evaluation_config", "conversation_count",
            "user_segment_count", "system_segment_count",
        )
    }
    summary["event_count"] = len(details["events"])
    summary["results"] = {
        metric: {field: value[field] for field in _RESULT_SUMMARY_FIELDS}
        for metric, value in details["results"].items()
    }
    return PipelineNodeResult(
        stage=result.stage,
        node_id=result.node_id,
        version=result.version,
        details=summary,
        internal_stages=result.internal_stages,
    )


def pipeline_id_for_metric(metric: str) -> str:
    """Return the atomic route identity for one registered timing metric."""

    normalized_metric = canonical_metric(metric)
    if normalized_metric not in _METRICS:
        raise ValueError(f"Unsupported FULL_DUPLEX metric: {metric}")
    return build_atomic_pipeline_id("full_duplex", "any", normalized_metric, _identity_components())


def _identity_components():
    return (
        node_component("normalization/duplex_timeline"),
        node_component("scoring/fdbench_timing"),
    )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
