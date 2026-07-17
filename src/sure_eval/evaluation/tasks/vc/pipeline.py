"""VC task routes built from transcription nodes and canonical ASR scoring."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import zip_longest
from statistics import fmean
from typing import Any

from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, MetricInputContract
from sure_eval.evaluation.nodes.transcription.common.audio_semantic import (
    TranscriptionRunner,
    asr_metric_for_semantic,
    default_semantic_metric,
    score_transcripts_with_asr,
    transcribe_audio,
    transcriber_for_language,
    uses_cer,
)
from sure_eval.evaluation.nodes.scoring._audio_quality_dispatch import (
    score_mos_metric,
    score_speaker_metric,
)
from sure_eval.evaluation.tasks.vc.types import VCSample

_ZH_DEFAULT_SEMANTIC_NORMALIZER = "punctuation_strip"
_VC_SEMANTIC_TEXT_CONTRACT = MetricInputContract(
    metric_id="semantic/asr_error_rate",
    required_roles=("converted_audio", "reference_text"),
    optional_roles=("reference_audio", "source_audio"),
    row_format="audio_with_inline_text",
    alignment_key="sample_id",
    aggregation="corpus_edit_distance",
    purpose="vc_content_preservation_via_asr_transcript",
)
_VC_SEMANTIC_AUDIO_CONTRACT = MetricInputContract(
    metric_id="semantic/asr_error_rate",
    required_roles=("converted_audio", "reference_audio"),
    optional_roles=("source_audio",),
    row_format="paired_audio_transcripts",
    alignment_key="sample_id",
    aggregation="corpus_edit_distance",
    purpose="vc_content_preservation_via_paired_asr_transcripts",
)
_ZIP_SENTINEL = object()


def evaluate_vc_samples(
    samples: list[VCSample],
    *,
    metrics: Iterable[str] | None = None,
    semantic_normalizer: str | None = None,
    transcribers: Mapping[str, TranscriptionRunner] | None = None,
    speaker_providers: Mapping[str, Any] | None = None,
    mos_providers: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Evaluate VC metrics through task-level pipeline nodes."""

    if not samples:
        raise ValueError("at least one VC sample is required")

    requested_metrics = [metric.lower() for metric in (metrics or _default_metrics(samples, prefix="vc"))]
    language = _common_language([sample.language for sample in samples])
    rows = [_base_row(sample) for sample in samples]
    results: dict[str, dict[str, Any]] = {}
    trace = []
    scoring_result: dict[str, Any] | None = None
    semantic_input_contract: MetricInputContract | None = None

    semantic_metrics = [metric for metric in requested_metrics if metric in {"vc_wer", "vc_cer"}]
    if semantic_metrics:
        if len(semantic_metrics) != 1:
            raise ValueError("VC task route supports one semantic metric per call")
        metric_name = semantic_metrics[0]
        expected_metric = default_semantic_metric("vc", language)
        if metric_name != expected_metric:
            raise ValueError(f"Metric {metric_name} does not match language {language}; expected {expected_metric}")
        effective_semantic_normalizer = _effective_semantic_normalizer(
            language=language,
            explicit_normalizer=semantic_normalizer,
        )
        semantic, semantic_input_contract = _evaluate_semantic(
            samples,
            rows,
            metric_name=metric_name,
            language=language,
            semantic_normalizer=effective_semantic_normalizer,
            transcribers=transcribers,
        )
        semantic_result = _semantic_metric_result(metric_name, semantic)
        results[metric_name] = semantic_result
        trace.extend(semantic.trace)
        scoring_result = semantic_result["asr_result"]

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

    unsupported = [
        metric
        for metric in requested_metrics
        if metric not in results and metric not in {"vc_wer", "vc_cer"} and not _is_speaker_metric(metric)
    ]
    if unsupported:
        raise ValueError(f"Unsupported VC metric(s): {', '.join(unsupported)}")
    if not results:
        raise ValueError("No VC metrics were evaluated")

    input_files = _input_files(
        samples,
        uses_reference_text=all(bool(sample.reference_text) for sample in samples),
    )
    metric = requested_metrics[0] if len(requested_metrics) == 1 else "multi"
    if metric in {"vc_wer", "vc_cer"} and len(results) == 1:
        input_contract = semantic_input_contract
        if input_contract is not None:
            input_contract.validate(input_files)
        transcript_node = "paraformer_zh" if uses_cer(language) else "whisper_large_v3"
        asr_metric = asr_metric_for_semantic(metric, language)
        normalizer_label = _normalizer_label_from_asr_pipeline(results[metric]["asr_pipeline_id"])
        if uses_cer(language):
            pipeline_id = f"vc.{language}.{metric}.funasr_loader_16k_mono.{transcript_node}.{normalizer_label}.wenet_{asr_metric}"
        else:
            pipeline_id = f"vc.{language}.{metric}.{transcript_node}.{normalizer_label}.wenet_{asr_metric}"
    else:
        input_contract = None
        pipeline_id = f"vc.{language}.multi.audio_metric_nodes"

    return EvaluationReport(
        task="VC",
        language=language,
        metric=metric,
        score=float(results[requested_metrics[0]]["score"]),
        pipeline_id=pipeline_id,
        pipeline_trace=tuple(trace),
        input_contract=input_contract,
        input_files=input_files,
        details={
            **({"scoring_result": scoring_result} if scoring_result is not None else {}),
            "results": results,
            "rows": rows,
            "input_contract": input_contract.as_dict() if input_contract else {},
            "input_files": input_files.as_dict(),
        },
    )


def _evaluate_semantic(
    samples: list[VCSample],
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    language: str,
    semantic_normalizer: str | None,
    transcribers: Mapping[str, TranscriptionRunner] | None,
):
    runner = transcriber_for_language(language, transcribers)
    references: list[str] = []
    hypotheses: list[str] = []
    keys: list[str] = []
    trace = []
    uses_reference_text = all(bool(sample.reference_text) for sample in samples)

    for index, sample in enumerate(samples, start=1):
        transcript, prediction_trace = transcribe_audio(
            sample.converted_audio,
            language=sample.language,
            runner=runner,
            role="converted_audio",
        )
        trace.extend(prediction_trace)
        reference_audio_transcript = ""
        reference_text = sample.reference_text
        if not reference_text:
            if not sample.reference_audio:
                raise ValueError("VC semantic evaluation requires reference_text or reference_audio")
            reference_audio_transcript, reference_trace = transcribe_audio(
                sample.reference_audio,
                language=sample.language,
                runner=runner,
                role="reference_audio",
            )
            trace.extend(reference_trace)
            reference_text = reference_audio_transcript

        sample_key = sample.sample_id or f"utt{index}"
        keys.append(sample_key)
        references.append(reference_text)
        hypotheses.append(transcript)
        rows[index - 1]["semantic"] = {
            "metric": metric_name,
            "transcript": transcript,
            "reference_text": reference_text,
            "reference_audio_transcript": reference_audio_transcript,
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
    input_contract = _VC_SEMANTIC_TEXT_CONTRACT if uses_reference_text else _VC_SEMANTIC_AUDIO_CONTRACT
    return semantic, input_contract


def _evaluate_speaker(
    samples: list[VCSample],
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    speaker_providers: Mapping[str, Any],
):
    backend_name = metric_name.removeprefix("sim/")
    provider = speaker_providers.get(backend_name)
    if provider is None:
        raise ValueError(f"VC speaker metric {metric_name} requires provider {backend_name}")
    speaker_rows = [
        (sample.sample_id or f"utt{index + 1}", sample.converted_audio, sample.reference_audio)
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
    samples: list[VCSample],
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    mos_providers: Mapping[str, Any],
):
    provider = mos_providers.get(metric_name)
    if provider is None:
        raise ValueError(f"VC MOS metric {metric_name} requires a provider")
    mos_rows = [
        (sample.sample_id or f"utt{index + 1}", sample.converted_audio)
        for index, sample in enumerate(samples)
    ]
    result = score_mos_metric(mos_rows, metric_name=metric_name, provider=provider)
    for row, per_sample in _zip_strict(rows, result.details["result"]["per_sample"]):
        row.setdefault("mos", {})[metric_name] = per_sample
    return result


def _default_metrics(samples: list[VCSample], *, prefix: str) -> tuple[str, ...]:
    return tuple(sorted({default_semantic_metric(prefix, sample.language) for sample in samples}))


def _effective_semantic_normalizer(*, language: str, explicit_normalizer: str | None) -> str | None:
    if explicit_normalizer is not None:
        return explicit_normalizer
    if uses_cer(language):
        return _ZH_DEFAULT_SEMANTIC_NORMALIZER
    return None


def _common_language(languages: list[str]) -> str:
    unique = {language for language in languages if language}
    if len(unique) != 1:
        raise ValueError("VC task route requires one language per call")
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


def _input_files(samples: list[VCSample], *, uses_reference_text: bool) -> EvaluationFiles:
    first = samples[0]
    roles = {
        "converted_audio": first.converted_audio if len(samples) == 1 else "batch",
    }
    if uses_reference_text:
        roles["reference_text"] = "inline"
        if first.reference_audio:
            roles["reference_audio"] = first.reference_audio if len(samples) == 1 else "batch"
    else:
        roles["reference_audio"] = first.reference_audio if len(samples) == 1 else "batch"
    if first.source_audio:
        roles["source_audio"] = first.source_audio if len(samples) == 1 else "batch"
    return EvaluationFiles(roles=roles)


def _base_row(sample: VCSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "converted_audio": sample.converted_audio,
        "reference_audio": sample.reference_audio,
        "source_audio": sample.source_audio,
        "language": sample.language,
        "metadata": dict(sample.metadata),
    }


def _is_speaker_metric(metric_name: str) -> bool:
    return metric_name in {"sim/wavlm-large", "sim/ecapa-tdnn", "sim/eres2net"}
