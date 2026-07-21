"""Speech enhancement task routes built from reusable audio scoring nodes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import zip_longest
from typing import Any

from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, MetricInputContract
from sure_eval.evaluation.nodes.scoring._audio_quality_dispatch import (
    score_full_reference_metric,
    score_mos_metric,
)
from sure_eval.evaluation.nodes.scoring._full_reference_audio import PESQProvider, SISDRProvider, STOIProvider
from sure_eval.evaluation.tasks.se.types import SESample

_ZIP_SENTINEL = object()
_MOS_METRICS = {"dnsmos", "wv-mos", "utmos"}
_FULL_REFERENCE_METRICS = {"si-sdr", "stoi", "pesq"}
_DEFAULT_METRICS = ("si-sdr", "stoi", "pesq", "dnsmos", "wv-mos", "utmos")
_SINGLE_PIPELINE_IDS = {
    "si-sdr": "se.si_sdr.si_sdr",
    "stoi": "se.stoi.stoi",
    "pesq": "se.pesq.pesq",
    "dnsmos": "se.dnsmos.dnsmos",
    "wv-mos": "se.wv_mos.wv_mos",
    "utmos": "se.utmos.utmos",
}

_FULL_REFERENCE_CONTRACT = MetricInputContract(
    metric_id="scoring/full_reference_audio",
    required_roles=("enhanced_audio", "reference_audio"),
    optional_roles=("noisy_audio",),
    row_format="speech_enhancement_audio_pairs",
    alignment_key="sample_id",
    aggregation="mean",
    purpose="se_full_reference_quality",
)
_NO_REFERENCE_CONTRACT = MetricInputContract(
    metric_id="scoring/no_reference_audio_quality",
    required_roles=("enhanced_audio",),
    optional_roles=("noisy_audio", "reference_audio"),
    row_format="enhanced_audio_rows",
    alignment_key="sample_id",
    aggregation="mean",
    purpose="se_no_reference_quality",
)


def evaluate_se_samples(
    samples: list[SESample],
    *,
    metrics: Iterable[str] | None = None,
    mos_providers: Mapping[str, Any] | None = None,
    reference_providers: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Evaluate speech enhancement metrics through task-level scoring nodes."""

    if not samples:
        raise ValueError("at least one SE sample is required")

    requested_metrics = tuple(_normalize_metric(metric) for metric in (metrics or _DEFAULT_METRICS))
    unsupported = [metric for metric in requested_metrics if metric not in _MOS_METRICS | _FULL_REFERENCE_METRICS]
    if unsupported:
        raise ValueError(f"Unsupported SE metric(s): {', '.join(unsupported)}")

    rows = [_base_row(sample) for sample in samples]
    results: dict[str, dict[str, Any]] = {}
    trace = []

    reference_providers = dict(reference_providers or {})
    for metric_name in [metric for metric in requested_metrics if metric in _FULL_REFERENCE_METRICS]:
        full_reference_result = _evaluate_full_reference(
            samples,
            rows,
            metric_name=metric_name,
            reference_providers=reference_providers,
        )
        results[metric_name] = full_reference_result.details["result"]
        trace.append(full_reference_result)

    mos_providers = dict(mos_providers or {})
    for metric_name in [metric for metric in requested_metrics if metric in _MOS_METRICS]:
        mos_result = _evaluate_mos(
            samples,
            rows,
            metric_name=metric_name,
            mos_providers=mos_providers,
        )
        results[metric_name] = mos_result.details["result"]
        trace.append(mos_result)

    if not results:
        raise ValueError("No SE metrics were evaluated")

    input_files = _input_files(samples)
    metric = requested_metrics[0] if len(requested_metrics) == 1 else "multi"
    input_contract = _input_contract_for_metrics(requested_metrics)
    input_contract.validate(input_files)
    pipeline_id = _SINGLE_PIPELINE_IDS[metric] if len(requested_metrics) == 1 else "se.multi.enhancement_quality_nodes"

    return EvaluationReport(
        task="SE",
        language="n/a",
        metric=metric,
        score=float(results[requested_metrics[0]]["score"]),
        pipeline_id=pipeline_id,
        pipeline_trace=tuple(trace),
        input_contract=input_contract,
        input_files=input_files,
        details={
            "results": results,
            "rows": rows,
            "input_contract": input_contract.as_dict(),
            "input_files": input_files.as_dict(),
        },
    )


def _evaluate_full_reference(
    samples: list[SESample],
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    reference_providers: Mapping[str, Any],
):
    provider = reference_providers.get(metric_name) or _default_reference_provider(metric_name)
    missing_references = [
        sample.sample_id or f"utt{index + 1}"
        for index, sample in enumerate(samples)
        if not sample.reference_audio
    ]
    if missing_references:
        preview = ", ".join(missing_references[:5])
        if len(missing_references) > 5:
            preview = f"{preview}, ..."
        raise ValueError(
            f"SE metric {metric_name} requires reference_audio for every sample; missing: {preview}"
        )
    scoring_rows = []
    row_indexes = []
    for index, sample in enumerate(samples):
        scoring_rows.append((sample.sample_id or f"utt{index + 1}", sample.enhanced_audio, sample.reference_audio))
        row_indexes.append(index)
    result = score_full_reference_metric(scoring_rows, metric_name=metric_name, provider=provider)
    for row_index, per_sample in _zip_strict(row_indexes, result.details["result"]["per_sample"]):
        rows[row_index].setdefault("full_reference", {})[metric_name] = per_sample
    return result


def _evaluate_mos(
    samples: list[SESample],
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    mos_providers: Mapping[str, Any],
):
    provider = mos_providers.get(metric_name)
    if provider is None:
        raise ValueError(f"SE MOS metric {metric_name} requires a provider")
    mos_rows = [
        (sample.sample_id or f"utt{index + 1}", sample.enhanced_audio)
        for index, sample in enumerate(samples)
    ]
    result = score_mos_metric(mos_rows, metric_name=metric_name, provider=provider)
    for row, per_sample in _zip_strict(rows, result.details["result"]["per_sample"]):
        row.setdefault("mos", {})[metric_name] = per_sample
    return result


def _default_reference_provider(metric_name: str):
    if metric_name == "si-sdr":
        return SISDRProvider()
    if metric_name == "stoi":
        return STOIProvider()
    if metric_name == "pesq":
        return PESQProvider()
    raise ValueError(f"Unsupported SE full-reference metric: {metric_name}")


def _input_contract_for_metrics(metrics: tuple[str, ...]) -> MetricInputContract:
    if any(metric in _FULL_REFERENCE_METRICS for metric in metrics):
        return _FULL_REFERENCE_CONTRACT
    return _NO_REFERENCE_CONTRACT


def _input_files(samples: list[SESample]) -> EvaluationFiles:
    first = samples[0]
    roles = {
        "enhanced_audio": first.enhanced_audio if len(samples) == 1 else "batch",
    }
    if first.noisy_audio:
        roles["noisy_audio"] = first.noisy_audio if len(samples) == 1 else "batch"
    if first.reference_audio:
        roles["reference_audio"] = first.reference_audio if len(samples) == 1 else "batch"
    return EvaluationFiles(roles=roles)


def _base_row(sample: SESample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "enhanced_audio": sample.enhanced_audio,
        "noisy_audio": sample.noisy_audio,
        "reference_audio": sample.reference_audio,
        "language": sample.language,
        "metadata": dict(sample.metadata),
    }


def _normalize_metric(metric: str) -> str:
    normalized = str(metric).strip().lower().replace("_", "-")
    aliases = {
        "sisdr": "si-sdr",
        "si-sdr": "si-sdr",
        "wvmos": "wv-mos",
        "wv-mos": "wv-mos",
        "dnsmos": "dnsmos",
        "utmos": "utmos",
        "stoi": "stoi",
        "pesq": "pesq",
    }
    return aliases.get(normalized, normalized)


def _zip_strict(*iterables):
    for values in zip_longest(*iterables, fillvalue=_ZIP_SENTINEL):
        if any(value is _ZIP_SENTINEL for value in values):
            raise ValueError("zip() argument lengths differ")
        yield values
