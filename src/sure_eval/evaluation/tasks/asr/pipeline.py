"""ASR task routes built from reusable pipeline nodes."""

from __future__ import annotations

from pathlib import Path

from sure_eval.evaluation.core.pipeline import run_pipeline
from sure_eval.evaluation.core.types import (
    EvaluationFiles,
    EvaluationReport,
    KeyTextFiles,
    MetricInputContract,
    PipelineNodeResult,
    PipelineSpec,
)
from sure_eval.evaluation.nodes.normalization.aispeech_norm import (
    normalize_asr_files,
    normalize_codeswitch_asr_files,
)
from sure_eval.evaluation.nodes.normalization.canonical_itn import normalize_canonical_asr_files
from sure_eval.evaluation.nodes.normalization.whisper_norm import normalize_whisper_asr_files
from sure_eval.evaluation.nodes.normalization.wetext_norm import (
    SUPPORTED_PROFILES as WETEXT_SUPPORTED_PROFILES,
    normalize_wetext_key_text_files,
)
from sure_eval.evaluation.nodes.scoring.wenet_wer import (
    score_codeswitch_mer,
    score_wenet_cer,
    score_wenet_wer,
)
from sure_eval.evaluation.nodes.scoring.sctk_sclite import (
    score_sctk_sclite_cer,
    score_sctk_sclite_wer,
)
from sure_eval.evaluation.nodes.scoring.token_cer import score_token_cer
from sure_eval.evaluation.nodes.scoring.token_mer import score_token_mer

_ASR_CONTRACTS = {
    "cer": MetricInputContract(
        metric_id="scoring/wenet_cer",
        required_roles=("hyp", "ref"),
        aggregation="corpus_edit_distance",
        purpose="character_error_rate",
    ),
    "wer": MetricInputContract(
        metric_id="scoring/wenet_wer",
        required_roles=("hyp", "ref"),
        aggregation="corpus_edit_distance",
        purpose="word_error_rate",
    ),
    "mer": MetricInputContract(
        metric_id="scoring/wenet_mer",
        required_roles=("hyp", "ref"),
        aggregation="corpus_edit_distance",
        purpose="mixed_error_rate_for_code_switch_asr",
    ),
    "cer_canonical": MetricInputContract(
        metric_id="scoring/token_cer",
        required_roles=("hyp", "ref"),
        aggregation="corpus_token_edit_distance",
        purpose="canonical_written_form_character_error_rate",
    ),
    "mer_canonical": MetricInputContract(
        metric_id="scoring/token_mer",
        required_roles=("hyp", "ref"),
        aggregation="corpus_token_edit_distance",
        purpose="canonical_mixed_token_error_rate",
    ),
    "wer_canonical": MetricInputContract(
        metric_id="scoring/token_mer",
        required_roles=("hyp", "ref"),
        aggregation="corpus_token_edit_distance",
        purpose="canonical_written_form_word_error_rate",
    ),
}

_CANONICAL_METRICS = {"cer_canonical", "mer_canonical", "wer_canonical"}


def evaluate_asr_files(
    ref_file: str,
    hyp_file: str,
    *,
    language: str,
    metric: str,
    normalizer: str | None = None,
    scorer: str | None = None,
) -> EvaluationReport:
    """Evaluate ASR key-text files through the configured task pipeline."""

    normalized_metric = _normalize_metric(language=language, metric=metric)
    normalized_scorer = _normalize_scorer(language=language, metric=normalized_metric, scorer=scorer)
    input_files = EvaluationFiles.from_ref_hyp(ref_file=ref_file, hyp_file=hyp_file)
    _ASR_CONTRACTS[normalized_metric].validate(input_files)
    if language == "cs" and normalized_metric not in _CANONICAL_METRICS:
        if normalized_metric != "mer":
            raise ValueError(f"Unsupported code-switch ASR metric: {normalized_metric}")
        if normalized_scorer != "wenet":
            raise ValueError("ASR code-switch MER does not support explicit scorer selection")
        if normalizer:
            raise ValueError("ASR code-switch MER does not support explicit normalizer selection")
        return _evaluate_codeswitch(ref_file, hyp_file, metric=normalized_metric, input_files=input_files)
    normalized_normalizer = _normalize_normalizer(
        language=language,
        metric=normalized_metric,
        normalizer=normalizer,
    )
    if normalized_metric in {"cer", "wer"} | _CANONICAL_METRICS:
        return _evaluate_regular(
            ref_file,
            hyp_file,
            language=language,
            metric=normalized_metric,
            normalizer=normalized_normalizer,
            scorer=normalized_scorer,
            input_files=input_files,
        )
    raise ValueError(f"Unsupported ASR metric for language={language}: {metric}")


def _evaluate_regular(
    ref_file: str,
    hyp_file: str,
    *,
    language: str,
    metric: str,
    normalizer: str,
    scorer: str,
    input_files: EvaluationFiles,
) -> EvaluationReport:
    input_contract = _ASR_CONTRACTS[metric]
    normalizer_node, normalizer_label = _normalization_node(language=language, normalizer=normalizer)
    scoring_node, score_label = _scoring_node(metric=metric, scorer=scorer)
    spec = PipelineSpec(
        pipeline_id=f"asr.{language}.{metric}.{normalizer_label}.{score_label}",
        task="ASR",
        language=language,
        metric=metric,
        nodes=(
            normalizer_node,
            scoring_node,
        ),
    )
    try:
        _, trace = run_pipeline(spec, KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file))
        scoring_result = trace[-1].details["result"]
        return EvaluationReport(
            task="ASR",
            language=language,
            metric=metric,
            score=float(scoring_result["score"]),
            pipeline_id=spec.pipeline_id,
            pipeline_trace=trace,
            input_contract=input_contract,
            input_files=input_files,
            details={
                "scoring_result": scoring_result,
                "input_contract": input_contract.as_dict(),
                "input_files": input_files.as_dict(),
            },
        )
    finally:
        _cleanup_trace_temp_files(locals().get("trace", ()))


def _evaluate_codeswitch(
    ref_file: str,
    hyp_file: str,
    *,
    metric: str,
    input_files: EvaluationFiles,
) -> EvaluationReport:
    if metric != "mer":
        raise ValueError(f"Unsupported code-switch ASR metric: {metric}")

    input_contract = _ASR_CONTRACTS[metric]
    trace: tuple[PipelineNodeResult, ...] = ()
    try:
        normalized_files, norm_result = normalize_codeswitch_asr_files(
            KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file)
        )
        _, score_result = score_codeswitch_mer(
            normalized_files,
            normalization_details=norm_result.details,
        )
        trace = (norm_result, score_result)
        scoring_result = score_result.details["result"]
        return EvaluationReport(
            task="ASR",
            language="cs",
            metric="mer",
            score=float(scoring_result["score"]),
            pipeline_id="asr.cs.mer.aispeech_norm.wenet_mer",
            pipeline_trace=trace,
            input_contract=input_contract,
            input_files=input_files,
            details={
                "scoring_result": scoring_result,
                "input_contract": input_contract.as_dict(),
                "input_files": input_files.as_dict(),
            },
        )
    finally:
        _cleanup_trace_temp_files(trace)


def _normalize_metric(*, language: str, metric: str) -> str:
    normalized = metric.lower()
    if language == "zh" and normalized == "wer":
        return "cer"
    if language == "cs" and normalized in {"wer", "cer"}:
        return "mer"
    return normalized


def _normalize_normalizer(*, language: str, metric: str, normalizer: str | None) -> str:
    normalized = (normalizer or "").lower().strip()
    if metric in _CANONICAL_METRICS:
        if normalized in {"", "canonical", "canonical_itn", "normalization/canonical_itn"}:
            return "canonical"
        raise ValueError(
            f"ASR {metric} requires the canonical_itn normalizer, got {normalizer!r}"
        )
    if normalized in {"canonical", "canonical_itn", "normalization/canonical_itn"}:
        raise ValueError("normalization/canonical_itn requires a canonical-family metric")
    if not normalized:
        if language == "en" and metric == "wer":
            return "whisper"
        return "aispeech"
    if normalized.startswith("wetext:"):
        profile = normalized.split(":", 1)[1]
        _validate_wetext_profile_for_language(language=language, profile=profile)
        return f"wetext:{profile}"
    if normalized in {"whisper", "whisper_norm", "normalization/whisper_norm"}:
        if language != "en" or metric != "wer":
            raise ValueError("whisper_norm is only a default-supported normalizer for English WER")
        return "whisper"
    if normalized in {"aispeech", "aispeech_norm", "normalization/aispeech_norm"}:
        return "aispeech"
    raise ValueError(f"Unsupported ASR normalizer: {normalizer}")


def _normalize_scorer(*, language: str, metric: str, scorer: str | None) -> str:
    normalized = (scorer or "").lower().strip().replace("-", "_")
    if metric == "cer_canonical":
        if normalized in {"", "token", "token_cer", "scoring/token_cer"}:
            return "token"
        raise ValueError(f"ASR cer_canonical only supports the token_cer scorer, got {scorer!r}")
    if metric in {"mer_canonical", "wer_canonical"}:
        if normalized in {"", "token", "token_mer", "scoring/token_mer"}:
            return "token_mer"
        raise ValueError(f"ASR {metric} only supports the token_mer scorer, got {scorer!r}")
    if normalized in {"token", "token_cer", "scoring/token_cer", "token_mer", "scoring/token_mer"}:
        raise ValueError("token scorers require a canonical-family metric")
    if not normalized:
        return "wenet"
    if normalized in {"wenet", "wenet_wer", "wenet_cer", "scoring/wenet_wer", "scoring/wenet_cer"}:
        return "wenet"
    if normalized in {"sctk", "sclite", "sctk_sclite", "scoring/sctk_sclite"}:
        if language == "cs":
            return "sctk_sclite"
        if metric not in {"wer", "cer"}:
            raise ValueError(f"sctk_sclite does not support ASR metric={metric!r}")
        return "sctk_sclite"
    raise ValueError(f"Unsupported ASR scorer: {scorer}")


def _normalization_node(*, language: str, normalizer: str):
    if normalizer == "canonical":
        return (
            lambda files: normalize_canonical_asr_files(files, language=language),
            "canonical_itn",
        )
    if normalizer.startswith("wetext:"):
        profile = normalizer.split(":", 1)[1]
        return (
            lambda files: normalize_wetext_key_text_files(files, profile=profile),
            f"wetext_{profile}",
        )
    if normalizer == "whisper":
        return (
            lambda files: normalize_whisper_asr_files(
                files,
                language=language,
                profile="english",
            ),
            "whisper_norm",
        )
    if normalizer == "aispeech":
        return (
            lambda files: normalize_asr_files(files, language=language),
            "aispeech_norm",
        )
    raise ValueError(f"Unsupported ASR normalizer: {normalizer}")


def _scoring_node(*, metric: str, scorer: str):
    if scorer == "token" and metric == "cer_canonical":
        return score_token_cer, "token_cer"
    if scorer == "token_mer" and metric in {"mer_canonical", "wer_canonical"}:
        return (
            lambda files: score_token_mer(files, metric=metric),
            "token_mer",
        )
    if scorer == "wenet":
        if metric == "cer":
            return score_wenet_cer, "wenet_cer"
        if metric == "wer":
            return score_wenet_wer, "wenet_wer"
    if scorer == "sctk_sclite":
        if metric == "cer":
            return score_sctk_sclite_cer, "sctk_sclite_cer"
        if metric == "wer":
            return score_sctk_sclite_wer, "sctk_sclite_wer"
    raise ValueError(f"Unsupported ASR scorer {scorer!r} for metric={metric!r}")


def _validate_wetext_profile_for_language(*, language: str, profile: str) -> None:
    if profile not in WETEXT_SUPPORTED_PROFILES:
        supported = ", ".join(sorted(WETEXT_SUPPORTED_PROFILES))
        raise ValueError(f"Unsupported wetext_norm profile {profile!r}; supported: {supported}")
    language_family = _wetext_language_family(language)
    if language_family is None:
        raise ValueError(f"wetext_norm is not mapped for ASR language={language!r}")
    if not profile.startswith(f"{language_family}_"):
        raise ValueError(f"wetext_norm profile {profile!r} does not match ASR language={language!r}")


def _wetext_language_family(language: str) -> str | None:
    normalized = language.lower().replace("_", "-")
    if normalized in {"zh", "zh-cn", "cmn"}:
        return "zh"
    if normalized in {"en", "en-us", "en-gb"}:
        return "en"
    if normalized in {"ja", "jp", "jpn"}:
        return "ja"
    return None


def _cleanup_trace_temp_files(trace: tuple[PipelineNodeResult, ...]) -> None:
    for result in trace:
        for key in ("ref_file", "hyp_file"):
            value = result.details.get(key)
            if isinstance(value, str):
                Path(value).unlink(missing_ok=True)
        side_outputs = result.details.get("side_outputs")
        if isinstance(side_outputs, dict):
            for value in side_outputs.values():
                if isinstance(value, str):
                    Path(value).unlink(missing_ok=True)
