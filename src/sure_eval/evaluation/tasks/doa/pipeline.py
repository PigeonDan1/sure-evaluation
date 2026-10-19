"""Model-independent canonical DOA task routes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, MetricInputContract
from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization
from sure_eval.evaluation.pipeline_identity import (
    build_atomic_pipeline_id,
    component_trace_ids,
    node_component,
)

_CANONICAL_CONTRACT = MetricInputContract(
    metric_id="task/doa_jsonl",
    required_roles=("reference_jsonl", "sample_output"),
    optional_roles=(
        "dimension",
        "threshold_deg",
        "aggregation",
        "timestamp_tolerance_sec",
        "prediction_activity_policy",
    ),
    row_format="doa_frame_sources_jsonl",
    alignment_key="recording_id_frame_index",
    aggregation="micro_matched_pair_or_macro_recording",
    purpose="canonical_direction_of_arrival_localization",
)

_CANONICAL_METRICS = {
    "mae",
    "rmse",
    "acc_10",
    "acc_15",
    "acc_30",
    "localization_recall",
    "localization_precision",
    "localization_f1",
}


def evaluate_doa_files(
    *,
    reference_jsonl: str | Path | None = None,
    sample_output: str | Path | None = None,
    dimension: str = "2d",
    threshold_deg: float = 10.0,
    aggregation: str = "micro",
    timestamp_tolerance_sec: float = 1e-6,
    prediction_activity_policy: str = "presence",
    metric: str = "mae",
) -> EvaluationReport:
    normalized_metric = metric.lower().replace("-", "_")
    if not reference_jsonl or not sample_output:
        raise ValueError("DOA route requires reference_jsonl and sample_output")
    if normalized_metric not in _CANONICAL_METRICS:
        raise ValueError(f"Unsupported DOA metric: {metric}")
    contract = _CANONICAL_CONTRACT
    input_files = EvaluationFiles(
        roles={
            "reference_jsonl": str(reference_jsonl),
            "sample_output": str(sample_output),
        }
    )
    contract.validate(input_files)
    bundle, normalization_result = normalize_doa_jsonl(
        reference_jsonl,
        sample_output,
        dimension=dimension,
        timestamp_tolerance_sec=timestamp_tolerance_sec,
        prediction_activity_policy=prediction_activity_policy,
    )
    scoring_result = score_doa_localization(
        bundle,
        metric=normalized_metric,
        threshold_deg=threshold_deg,
        aggregation=aggregation,
    )
    components = (
        node_component("normalization/doa_jsonl"),
        node_component("scoring/doa_localization"),
    )
    details = {
        "route": "canonical_doa",
        "evaluation_config": {
            "dimension": dimension,
            "threshold_deg": scoring_result.details["threshold_deg"],
            "configured_threshold_deg": threshold_deg,
            "aggregation": aggregation,
            "timestamp_tolerance_sec": timestamp_tolerance_sec,
            "prediction_activity_policy": prediction_activity_policy,
        },
        "input_sha256": {role: _sha256_file(path) for role, path in input_files.roles.items()},
        "normalization": bundle.as_dict(),
        "scoring": scoring_result.details,
    }
    pipeline_id = build_atomic_pipeline_id("doa", "any", normalized_metric, components)
    score = scoring_result.details.get("score")
    return EvaluationReport(
        task="DOA",
        language="n/a",
        metric=normalized_metric,
        score=None if score is None else float(score),
        pipeline_id=pipeline_id,
        pipeline_trace=(normalization_result, scoring_result),
        input_contract=contract,
        input_files=input_files,
        computation_node_ids=component_trace_ids(components),
        details=details,
    )


def pipeline_id_for_metric(metric: str) -> str:
    normalized = metric.lower().replace("-", "_")
    if normalized not in _CANONICAL_METRICS:
        raise ValueError(f"Unsupported DOA metric: {metric}")
    components = (
        node_component("normalization/doa_jsonl"),
        node_component("scoring/doa_localization"),
    )
    return build_atomic_pipeline_id("doa", "any", normalized, components)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
