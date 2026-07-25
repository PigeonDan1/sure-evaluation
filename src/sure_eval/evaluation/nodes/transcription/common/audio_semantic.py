"""Shared helpers for audio tasks that score semantics through ASR."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from sure_eval.evaluation.core.types import EvaluationReport, PipelineNodeResult
from sure_eval.evaluation.nodes.frontend.funasr_loader_16k_mono import describe_funasr_loader_16k_mono
from sure_eval.evaluation.nodes.transcription.paraformer_zh import transcribe_paraformer_zh
from sure_eval.evaluation.nodes.transcription.qwen3_asr_1_7b import transcribe_qwen3_asr_1_7b
from sure_eval.evaluation.nodes.transcription.whisper_large_v3 import transcribe_whisper_large_v3
from sure_eval.evaluation.pipeline_identity import PipelineComponent, node_component
from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files


class TranscriptionRunner(Protocol):
    def transcribe(self, audio_path: str, *, language: str = "en") -> str:
        """Transcribe one audio file."""
        ...


@dataclass(frozen=True)
class SemanticASRBatchResult:
    score: float
    asr_metric: str
    asr_report: EvaluationReport
    trace: tuple[PipelineNodeResult, ...]
    rows: list[dict[str, Any]]


def uses_cer(language: str) -> bool:
    return language.lower().startswith(("zh", "cmn", "yue"))


def default_semantic_metric(prefix: str, language: str) -> str:
    return f"{prefix}_{'cer' if uses_cer(language) else 'wer'}"


def asr_metric_for_semantic(metric: str, language: str) -> str:
    normalized = metric.lower()
    if normalized.endswith("_cer") or normalized == "cer":
        return "cer"
    if normalized.endswith("_wer") or normalized == "wer":
        return "wer"
    return "cer" if uses_cer(language) else "wer"


def semantic_pipeline_components(
    language: str,
    asr_report: EvaluationReport,
    *,
    transcription_passes: int = 1,
    transcription_node_id: str | None = None,
) -> tuple[PipelineComponent, ...]:
    """Return logical audio-semantic components for a report-level pipeline ID."""

    components: list[PipelineComponent] = []
    for _ in range(transcription_passes):
        components.extend(_transcription_components(language, transcription_node_id))
    components.extend(_asr_trace_components(asr_report.pipeline_trace))
    return tuple(components)


def semantic_trace_components(trace: tuple[PipelineNodeResult, ...]) -> tuple[PipelineComponent, ...]:
    """Return identity components from the actual semantic trace."""

    return tuple(node_component(node.node_id, profile=_profile_for_asr_node(node)) for node in trace)


def transcriber_for_language(
    language: str,
    transcribers: Mapping[str, TranscriptionRunner] | None,
) -> TranscriptionRunner | None:
    if not transcribers:
        return None
    if language in transcribers:
        return transcribers[language]
    family = "zh" if uses_cer(language) else "en"
    return transcribers.get(family)


def transcription_node_needs_frontend(node_id: str, language: str) -> bool:
    """Return whether a transcription route includes the FunASR loader frontend."""

    return node_id == "transcription/paraformer_zh" or (not node_id and uses_cer(language))


def _asr_trace_components(trace: tuple[PipelineNodeResult, ...]) -> tuple[PipelineComponent, ...]:
    components: list[PipelineComponent] = []
    for node in trace:
        profile = _profile_for_asr_node(node)
        components.append(node_component(node.node_id, profile=profile))
    return tuple(components)


def _profile_for_asr_node(node: PipelineNodeResult) -> str | None:
    if node.node_id == "normalization/wetext_norm":
        return str(node.details.get("profile") or "")
    if node.node_id == "normalization/whisper_norm":
        return str(node.details.get("profile") or "english")
    if node.node_id == "normalization/aispeech_norm":
        return str(node.details.get("profile") or "")
    if node.node_id == "normalization/canonical_itn":
        profile = str(node.details.get("profile") or node.details.get("language") or "")
        return profile.removesuffix("_canonical")
    return None


def transcribe_audio(
    audio_path: str,
    *,
    language: str,
    runner: TranscriptionRunner | None,
    role: str,
    transcription_node_id: str | None = None,
) -> tuple[str, tuple[PipelineNodeResult, ...]]:
    selected_node = transcription_node_id or _default_transcription_node_id(language)
    if selected_node == "transcription/qwen3_asr_1_7b":
        transcript, transcription_result = transcribe_qwen3_asr_1_7b(
            audio_path,
            language=language,
            runner=runner,
            role=role,
        )
        return transcript, (transcription_result,)
    if selected_node == "transcription/paraformer_zh":
        frontend_result = describe_funasr_loader_16k_mono(
            audio_path,
            language=language,
            role=role,
        )
        transcript, transcription_result = transcribe_paraformer_zh(
            audio_path,
            language=language,
            runner=runner,
            role=role,
        )
        return transcript, (frontend_result, transcription_result)
    if selected_node != "transcription/whisper_large_v3":
        raise ValueError(f"Unsupported semantic transcription node: {selected_node}")
    transcript, transcription_result = transcribe_whisper_large_v3(
        audio_path,
        language=language,
        runner=runner,
        role=role,
    )
    return transcript, (transcription_result,)


def _transcription_components(
    language: str,
    transcription_node_id: str | None,
) -> tuple[PipelineComponent, ...]:
    selected_node = transcription_node_id or _default_transcription_node_id(language)
    if selected_node == "transcription/qwen3_asr_1_7b":
        return (node_component("transcription/qwen3_asr_1_7b"),)
    if selected_node == "transcription/paraformer_zh":
        return (
            node_component("frontend/funasr_loader_16k_mono"),
            node_component("transcription/paraformer_zh"),
        )
    if selected_node == "transcription/whisper_large_v3":
        return (node_component("transcription/whisper_large_v3"),)
    raise ValueError(f"Unsupported semantic transcription node: {selected_node}")


def _default_transcription_node_id(language: str) -> str:
    return "transcription/paraformer_zh" if uses_cer(language) else "transcription/whisper_large_v3"


def score_transcripts_with_asr(
    *,
    references: list[str],
    hypotheses: list[str],
    keys: list[str],
    language: str,
    asr_metric: str,
    normalizer: str | None = None,
    rows: list[dict[str, Any]],
    transcription_trace: list[PipelineNodeResult],
) -> SemanticASRBatchResult:
    if not references:
        raise ValueError("at least one semantic reference is required")
    if len(references) != len(hypotheses) or len(references) != len(keys):
        raise ValueError("references, hypotheses, and keys must have the same length")

    ref_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    hyp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        ref_path = ref_file.name
        hyp_path = hyp_file.name
        for key, reference in zip(keys, references, strict=True):
            ref_file.write(f"{key}\t{reference}\n")
        for key, hypothesis in zip(keys, hypotheses, strict=True):
            hyp_file.write(f"{key}\t{hypothesis}\n")
        ref_file.close()
        hyp_file.close()

        asr_report = evaluate_asr_files(
            ref_path,
            hyp_path,
            language=language,
            metric=asr_metric,
            normalizer=normalizer,
        )
    finally:
        ref_file.close()
        hyp_file.close()
        Path(ref_file.name).unlink(missing_ok=True)
        Path(hyp_file.name).unlink(missing_ok=True)

    return SemanticASRBatchResult(
        score=asr_report.score,
        asr_metric=asr_report.metric,
        asr_report=asr_report,
        trace=tuple(transcription_trace) + asr_report.pipeline_trace,
        rows=rows,
    )
