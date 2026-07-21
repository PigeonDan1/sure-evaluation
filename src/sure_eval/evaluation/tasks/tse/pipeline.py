"""TSE task routes built from signal scoring, speaker similarity, MOS, and ASR nodes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import zip_longest
from statistics import fmean
from typing import Any

from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, MetricInputContract
from sure_eval.evaluation.nodes.scoring._audio_quality_dispatch import (
    score_mos_metric,
    score_speaker_metric,
)
from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr
from sure_eval.evaluation.tasks.tse.types import TSESample

_TSE_SEMANTIC_TEXT_CONTRACT = MetricInputContract(
    metric_id="semantic/asr_error_rate",
    required_roles=("prediction_audio", "reference_text"),
    optional_roles=("reference_audio", "mixed_audio", "enrollment_audio"),
    row_format="audio_with_inline_text",
    alignment_key="sample_id",
    aggregation="corpus_edit_distance",
    purpose="tse_intelligibility_via_asr_transcript",
)
_ZIP_SENTINEL = object()


def evaluate_tse_samples(
    samples: list[TSESample],
    *,
    metrics: Iterable[str] | None = None,
    semantic_normalizer: str | None = None,
    transcribers: Mapping[str, Any] | None = None,
    speaker_providers: Mapping[str, Any] | None = None,
    mos_providers: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Evaluate TSE metrics through task-level pipeline nodes."""

    if not samples:
        raise ValueError("at least one TSE sample is required")

    requested_metrics = [metric.lower() for metric in (metrics or _default_metrics(samples))]
    language = _common_language([sample.language for sample in samples])
    rows = [_base_row(sample) for sample in samples]
    results: dict[str, dict[str, Any]] = {}
    trace = []

    signal_metrics = [metric for metric in requested_metrics if metric == "si_sdr"]
    if signal_metrics:
        si_sdr_result = _evaluate_si_sdr(samples, rows)
        results["si_sdr"] = si_sdr_result.details["result"]
        trace.append(si_sdr_result)

    speaker_providers = dict(speaker_providers or {})
    for metric_name in [metric for metric in requested_metrics if _is_speaker_metric(metric)]:
        speaker_result = _evaluate_speaker(
            samples,
            rows,
            metric_name=metric_name,
            speaker_providers=speaker_providers,
        )
        results[metric_name] = speaker_result.details["result"]
        trace.append(speaker_result)

    mos_providers = dict(mos_providers or {})
    for metric_name in [metric for metric in requested_metrics if metric in {"dnsmos", "wv-mos", "utmos"}]:
        mos_result = _evaluate_mos(
            samples,
            rows,
            metric_name=metric_name,
            mos_providers=mos_providers,
        )
        results[metric_name] = mos_result.details["result"]
        trace.append(mos_result)

    semantic_metrics = [metric for metric in requested_metrics if metric in {"tse_wer", "tse_cer"}]
    scoring_result: dict[str, Any] | None = None
    if semantic_metrics:
        if len(semantic_metrics) != 1:
            raise ValueError("TSE task route supports one semantic metric per call")
        metric_name = semantic_metrics[0]
        from sure_eval.evaluation.nodes.transcription.common.audio_semantic import (
            default_semantic_metric,
        )

        expected_metric = default_semantic_metric("tse", language)
        if metric_name != expected_metric:
            raise ValueError(
                f"Metric {metric_name} does not match language {language}; expected {expected_metric}"
            )
        semantic = _evaluate_semantic(
            samples,
            rows,
            metric_name=metric_name,
            language=language,
            semantic_normalizer=semantic_normalizer,
            transcribers=transcribers,
        )
        semantic_result = _semantic_metric_result(metric_name, semantic)
        results[metric_name] = semantic_result
        trace.extend(semantic.trace)
        scoring_result = semantic_result.get("asr_result")

    unsupported = [
        metric
        for metric in requested_metrics
        if metric not in results
        and metric not in {"tse_wer", "tse_cer"}
        and not _is_speaker_metric(metric)
        and metric != "si_sdr"
    ]
    if unsupported:
        raise ValueError(f"Unsupported TSE metric(s): {', '.join(unsupported)}")
    if not results:
        raise ValueError("No TSE metrics were evaluated")

    input_files = _input_files(samples)
    metric = requested_metrics[0] if len(requested_metrics) == 1 else "multi"

    if metric == "si_sdr" and len(results) == 1:
        pipeline_id = f"tse.{language}.si_sdr.si_sdr"
    elif metric in {"tse_wer", "tse_cer"} and len(results) == 1:
        from sure_eval.evaluation.nodes.transcription.common.audio_semantic import (
            asr_metric_for_semantic,
            uses_cer,
        )

        transcript_node = "paraformer_zh" if uses_cer(language) else "whisper_large_v3"
        asr_metric = asr_metric_for_semantic(metric, language)
        normalizer_label = _normalizer_label_from_asr_pipeline(results[metric]["asr_pipeline_id"])
        if uses_cer(language):
            pipeline_id = (
                f"tse.{language}.{metric}.funasr_loader_16k_mono."
                f"{transcript_node}.{normalizer_label}.wenet_{asr_metric}"
            )
        else:
            pipeline_id = f"tse.{language}.{metric}.{transcript_node}.{normalizer_label}.wenet_{asr_metric}"
        input_contract = _TSE_SEMANTIC_TEXT_CONTRACT
        input_contract.validate(input_files)
        return EvaluationReport(
            task="TSE",
            language=language,
            metric=metric,
            score=float(results[metric]["score"]),
            pipeline_id=pipeline_id,
            pipeline_trace=tuple(trace),
            input_contract=input_contract,
            input_files=input_files,
            details={
                "scoring_result": scoring_result,
                "results": results,
                "rows": rows,
                "input_contract": input_contract.as_dict(),
                "input_files": input_files.as_dict(),
            },
        )
    else:
        pipeline_id = f"tse.{language}.multi.audio_metric_nodes"

    first_metric = requested_metrics[0]
    first_score = results[first_metric]["score"]

    return EvaluationReport(
        task="TSE",
        language=language,
        metric=metric,
        score=float(first_score),
        pipeline_id=pipeline_id,
        pipeline_trace=tuple(trace),
        input_contract=None,
        input_files=input_files,
        details={
            **({"scoring_result": scoring_result} if scoring_result is not None else {}),
            "results": results,
            "rows": rows,
            "input_files": input_files.as_dict(),
        },
    )


def _evaluate_si_sdr(
    samples: list[TSESample],
    rows: list[dict[str, Any]],
) -> Any:
    signal_rows: list[tuple[str, str, str]] = []
    mixed_paths: list[str] = []
    for index, sample in enumerate(samples):
        if not sample.reference_audio:
            continue
        signal_rows.append(
            (sample.sample_id or f"utt{index + 1}", sample.prediction_audio, sample.reference_audio)
        )
        mixed_paths.append(sample.mixed_audio or "")
    if not signal_rows:
        raise ValueError("SI-SDR requires reference_audio for at least one sample")
    mixed_arg = mixed_paths if any(mixed_paths) else None
    result = score_si_sdr(signal_rows, mixed_paths=mixed_arg)
    for row, per_sample in _zip_strict(rows, result.details["result"]["per_sample"]):
        row["signal"] = {"si_sdr": per_sample}
    return result


def _evaluate_speaker(
    samples: list[TSESample],
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    speaker_providers: Mapping[str, Any],
):
    backend_name = metric_name.removeprefix("sim/")
    provider = speaker_providers.get(backend_name)
    if provider is None:
        raise ValueError(f"TSE speaker metric {metric_name} requires provider {backend_name}")
    speaker_rows = [
        (sample.sample_id or f"utt{index + 1}", sample.prediction_audio, sample.reference_audio)
        for index, sample in enumerate(samples)
        if sample.reference_audio
    ]
    if not speaker_rows:
        raise ValueError("speaker similarity requires reference_audio for at least one sample")
    result = score_speaker_metric(
        speaker_rows,
        backend_name=backend_name,
        provider=provider,
    )
    row_indexes = [index for index, sample in enumerate(samples) if sample.reference_audio]
    for row_index, per_sample in _zip_strict(row_indexes, result.details["result"]["per_sample"]):
        rows[row_index].setdefault("speaker", {})[backend_name] = per_sample
    return result


def _evaluate_mos(
    samples: list[TSESample],
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    mos_providers: Mapping[str, Any],
):
    provider = mos_providers.get(metric_name)
    if provider is None:
        raise ValueError(f"TSE MOS metric {metric_name} requires a provider")
    mos_rows = [
        (sample.sample_id or f"utt{index + 1}", sample.prediction_audio)
        for index, sample in enumerate(samples)
    ]
    result = score_mos_metric(mos_rows, metric_name=metric_name, provider=provider)
    for row, per_sample in _zip_strict(rows, result.details["result"]["per_sample"]):
        row.setdefault("mos", {})[metric_name] = per_sample
    return result


def _evaluate_semantic(
    samples: list[TSESample],
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    language: str,
    semantic_normalizer: str | None,
    transcribers: Mapping[str, Any] | None,
):
    from sure_eval.evaluation.nodes.transcription.common.audio_semantic import (
        asr_metric_for_semantic,
        score_transcripts_with_asr,
        transcribe_audio,
        transcriber_for_language,
    )

    runner = transcriber_for_language(language, transcribers)
    references: list[str] = []
    hypotheses: list[str] = []
    keys: list[str] = []
    trace = []

    for index, sample in enumerate(samples, start=1):
        if not sample.reference_text:
            raise ValueError("TSE semantic evaluation requires reference_text")
        transcript, prediction_trace = transcribe_audio(
            sample.prediction_audio,
            language=sample.language,
            runner=runner,
            role="prediction_audio",
        )
        trace.extend(prediction_trace)
        sample_key = sample.sample_id or f"utt{index}"
        keys.append(sample_key)
        references.append(sample.reference_text)
        hypotheses.append(transcript)
        rows[index - 1]["semantic"] = {
            "metric": metric_name,
            "transcript": transcript,
            "reference_text": sample.reference_text,
            "asr_metric": asr_metric_for_semantic(metric_name, sample.language),
            "normalizer": semantic_normalizer,
        }

    asr_metric = asr_metric_for_semantic(metric_name, language)
    semantic = score_transcripts_with_asr(
        references=references,
        hypotheses=hypotheses,
        keys=keys,
        language=language,
        asr_metric=asr_metric,
        normalizer=semantic_normalizer,
        rows=rows,
        transcription_trace=trace,
    )
    return semantic


def _default_metrics(samples: list[TSESample]) -> tuple[str, ...]:
    return ("si_sdr",)


def _common_language(languages: list[str]) -> str:
    unique = {language for language in languages if language}
    if len(unique) != 1:
        raise ValueError("TSE task route requires one language per call")
    return unique.pop()


def _zip_strict(*iterables):
    for values in zip_longest(*iterables, fillvalue=_ZIP_SENTINEL):
        if any(value is _ZIP_SENTINEL for value in values):
            raise ValueError("zip() argument lengths differ")
        yield values


def _semantic_metric_result(metric_name: str, semantic) -> dict[str, Any]:
    rows = semantic.rows
    score_key = "cer" if semantic.asr_metric == "cer" else "wer"
    return {
        "metric_name": metric_name,
        "score": semantic.score,
        score_key: semantic.score,
        "num_samples": len(rows),
        "aggregation": "corpus_edit_distance",
        "asr_metric": semantic.asr_metric,
        "asr_pipeline_id": semantic.asr_report.pipeline_id,
        "asr_result": semantic.asr_report.details["scoring_result"],
        "mean_sample_score": fmean([semantic.score]) if rows else 0.0,
    }


def _normalizer_label_from_asr_pipeline(pipeline_id: str) -> str:
    parts = pipeline_id.split(".")
    if len(parts) >= 5:
        return parts[3]
    return "unknown_norm"


def _input_files(samples: list[TSESample]) -> EvaluationFiles:
    first = samples[0]
    roles = {
        "prediction_audio": first.prediction_audio if len(samples) == 1 else "batch",
        "reference_audio": first.reference_audio if len(samples) == 1 else "batch",
    }
    if first.mixed_audio:
        roles["mixed_audio"] = first.mixed_audio if len(samples) == 1 else "batch"
    if first.enrollment_audio:
        roles["enrollment_audio"] = first.enrollment_audio if len(samples) == 1 else "batch"
    if any(sample.reference_text for sample in samples):
        roles["reference_text"] = "inline"
    return EvaluationFiles(roles=roles)


def _base_row(sample: TSESample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "prediction_audio": sample.prediction_audio,
        "reference_audio": sample.reference_audio,
        "mixed_audio": sample.mixed_audio,
        "enrollment_audio": sample.enrollment_audio,
        "reference_text": sample.reference_text,
        "language": sample.language,
        "metadata": dict(sample.metadata),
    }


def _is_speaker_metric(metric_name: str) -> bool:
    return metric_name in {"sim/wavlm-large", "sim/ecapa-tdnn", "sim/eres2net"}