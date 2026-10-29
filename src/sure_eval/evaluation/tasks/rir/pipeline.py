"""RIR estimation and spatial-audio evaluation pipeline."""

from __future__ import annotations

from collections.abc import Iterable

from sure_eval.evaluation.core.types import (
    EvaluationFiles,
    EvaluationReport,
    MetricInputContract,
    PipelineNodeResult,
)
from sure_eval.evaluation.pipeline_identity import (
    build_atomic_pipeline_id,
    build_bundle_pipeline_id,
    canonical_metric,
    node_component,
)
from sure_eval.evaluation.tasks.rir.metrics import (
    rir_c50_mae,
    rir_c80_mae,
    rir_doa_mae,
    rir_drr_mae,
    rir_edc_mae,
    rir_iacc_early_mae,
    rir_iacc_full_mae,
    rir_iacc_late_mae,
    rir_iacc_mae,
    rir_ild_mae,
    rir_ipd_mae,
    rir_itd_mae,
    rir_npm,
    rt30_mae,
    rt60_mae,
    validate_rir_samples,
)
from sure_eval.evaluation.tasks.rir.types import RIRSample

DEFAULT_METRICS = ("rir_npm", "rt30_mae", "rt60_mae", "rir_drr_mae", "rir_c50_mae", "rir_c80_mae", "rir_edc_mae")
METRIC_ALIASES = {
    "npm": "rir_npm",
    "rir_npm": "rir_npm",
    "iacc": "rir_iacc_mae",
    "rir_iacc": "rir_iacc_mae",
    "iacc_full": "rir_iacc_full_mae",
    "iacc_early": "rir_iacc_early_mae",
    "iacc_late": "rir_iacc_late_mae",
    "itd": "rir_itd_mae",
    "rir_itd": "rir_itd_mae",
}
METRICS = {
    "rir_npm": ("scoring/rir_npm", rir_npm),
    "rt30_mae": ("scoring/rt30_mae", rt30_mae),
    "rt60_mae": ("scoring/rt60_mae", rt60_mae),
    "rir_drr_mae": ("scoring/rir_drr_mae", rir_drr_mae),
    "rir_c50_mae": ("scoring/rir_c50_mae", rir_c50_mae),
    "rir_c80_mae": ("scoring/rir_c80_mae", rir_c80_mae),
    "rir_edc_mae": ("scoring/rir_edc_mae", rir_edc_mae),
    "rir_iacc_mae": ("scoring/rir_iacc_mae", rir_iacc_mae),
    "rir_iacc_full_mae": ("scoring/rir_iacc_full_mae", rir_iacc_full_mae),
    "rir_iacc_early_mae": ("scoring/rir_iacc_early_mae", rir_iacc_early_mae),
    "rir_iacc_late_mae": ("scoring/rir_iacc_late_mae", rir_iacc_late_mae),
    "rir_itd_mae": ("scoring/rir_itd_mae", rir_itd_mae),
    "rir_ild_mae": ("scoring/rir_ild_mae", rir_ild_mae),
    "rir_ipd_mae": ("scoring/rir_ipd_mae", rir_ipd_mae),
    "rir_doa_mae": ("scoring/rir_doa_mae", rir_doa_mae),
}
IACC_METRICS = frozenset({"rir_iacc_mae", "rir_iacc_full_mae", "rir_iacc_early_mae", "rir_iacc_late_mae"})
SPATIAL_METRICS = frozenset(
    {"rir_iacc_mae", "rir_iacc_full_mae", "rir_iacc_early_mae", "rir_iacc_late_mae", "rir_itd_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"}
)


def evaluate_rir_samples(
    samples: list[RIRSample],
    *,
    metrics: Iterable[str] | None = None,
    channel_mode: str = "auto",
) -> EvaluationReport:
    if not samples:
        raise ValueError("at least one RIR sample is required")
    requested = tuple(_normalize_metric(metric) for metric in (metrics or DEFAULT_METRICS))
    if not requested:
        raise ValueError("at least one RIR metric is required")
    unsupported = [metric for metric in requested if metric not in METRICS]
    if unsupported:
        raise ValueError(f"Unsupported RIR metric(s): {', '.join(unsupported)}")
    if any(metric in SPATIAL_METRICS for metric in requested):
        _validate_spatial_mode(channel_mode)
    input_rows = validate_rir_samples(samples, channel_mode=channel_mode)

    results = {}
    trace = []
    for metric in requested:
        node_id, scorer = METRICS[metric]
        result = scorer(samples)
        results[metric] = {"metric_name": metric, "score": result.score, **result.details}
        trace.append(
            PipelineNodeResult(
                stage="scoring",
                node_id=node_id,
                version="v1",
                details={"result": results[metric]},
            )
        )
    members = tuple(_pipeline_id(metric) for metric in requested)
    bundle = len(members) > 1
    pipeline_id = members[0] if not bundle else build_bundle_pipeline_id("rir", "any", members)
    row_format = "binaural_rir_samples_jsonl" if any(
        metric in IACC_METRICS for metric in requested
    ) else "multichannel_rir_samples_jsonl" if any(
        metric in SPATIAL_METRICS for metric in requested
    ) else "rir_samples_jsonl"
    contract = MetricInputContract(
        metric_id="rir_samples_jsonl",
        required_roles=("prediction_rir", "reference_rir"),
        optional_roles=("sample_rate", "channel_layout", "metadata"),
        row_format=row_format,
        alignment_key="sample_id",
        aggregation="per_sample_then_metric_mean",
        purpose="rir_estimation_and_spatial_audio_quality",
    )
    return EvaluationReport(
        task="RIR estimation",
        language="n/a",
        metric=canonical_metric(requested[0]) if not bundle else "multi",
        score=float(results[requested[0]]["score"]),
        pipeline_id=pipeline_id,
        pipeline_trace=tuple(trace),
        input_contract=contract,
        input_files=EvaluationFiles(
            roles={"prediction_rir": "batch", "reference_rir": "batch"}
        ),
        pipeline_kind="bundle" if bundle else "atomic",
        member_pipeline_ids=members if bundle else (),
        computation_node_ids=tuple(METRICS[metric][0] for metric in requested),
        details={
            "channel_mode": _normalize_channel_mode(channel_mode),
            "results": results,
            "rows": input_rows,
        },
    )


def _normalize_metric(metric: str) -> str:
    normalized = str(metric).strip().lower().replace("-", "_")
    return METRIC_ALIASES.get(normalized, normalized)


def _normalize_channel_mode(channel_mode: str) -> str:
    normalized = str(channel_mode).strip().lower().replace("-", "_")
    return {
        "mono": "single",
        "single_channel": "single",
        "multichannel": "multi",
        "multi_channel": "multi",
        "binaural_lr": "binaural",
    }.get(normalized, normalized)


def _validate_spatial_mode(channel_mode: str) -> None:
    normalized = _normalize_channel_mode(channel_mode)
    if normalized not in {"auto", "multi", "binaural"}:
        raise ValueError("spatial RIR metrics require the multi-channel chain")


def _pipeline_id(metric: str) -> str:
    node_id = METRICS[metric][0]
    return build_atomic_pipeline_id("rir", "any", metric, (node_component(node_id),))
