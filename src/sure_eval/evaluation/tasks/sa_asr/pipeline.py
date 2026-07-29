"""SA-ASR task route built on the generic MeetEval scoring node."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sure_eval.evaluation.conversion.sa_asr__cpwer.stm_to_txt import convert_stm_to_txt
from sure_eval.evaluation.conversion.sa_asr__cpwer.txt_to_stm import convert_txt_to_stm
from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, MetricInputContract
from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.gstar_norm import normalize_gstar_sa_asr_files
from sure_eval.evaluation.nodes.normalization.whisper_norm import normalize_whisper_asr_files
from sure_eval.evaluation.nodes.scoring.meeteval import score_meeteval
from sure_eval.evaluation.pipeline_identity import (
    build_atomic_pipeline_id,
    component_trace_ids,
    conversion_component,
    node_component,
)

CONVERSION_ID = "sa_asr__cpwer"

_SA_ASR_CONTRACT = MetricInputContract(
    metric_id="scoring/meeteval",
    required_roles=("hyp", "ref"),
    row_format="meeteval_annotation",
    alignment_key="session_id",
    aggregation="meeteval_combined_error_rate",
    purpose="speaker_attributed_asr_cpwer",
)


def evaluate_sa_asr_files(
    ref_file: str,
    hyp_file: str,
    *,
    metric: str = "cpwer",
    language: str = "en",
    collar: float = 0.5,
    companion_metrics: tuple[str, ...] = ("der",),
    normalization_node: str | None = None,
    conversion_output_dir: str | None = None,
) -> EvaluationReport:
    """Evaluate speaker-attributed ASR annotations with MeetEval cpWER."""

    normalized_metric = metric.lower().replace("-", "_")
    if normalized_metric != "cpwer":
        raise ValueError(f"Unsupported SA-ASR metric: {metric}")
    input_files = EvaluationFiles.from_ref_hyp(ref_file=ref_file, hyp_file=hyp_file)
    _SA_ASR_CONTRACT.validate(input_files)
    trace: tuple[PipelineNodeResult, ...] = ()
    temp_paths: list[str] = []
    conversion_dir = Path(conversion_output_dir) if conversion_output_dir else None
    resolved_normalization_node = _resolve_normalization_node(
        language=language,
        normalization_node=normalization_node,
    )
    components = (
        conversion_component(CONVERSION_ID),
        _normalization_component(resolved_normalization_node),
        node_component("scoring/meeteval"),
    )
    pipeline_id = build_atomic_pipeline_id("sa_asr", language, "cpwer", components)
    if conversion_dir is not None:
        conversion_dir.mkdir(parents=True, exist_ok=True)
    try:
        ref_txt = _conversion_path(conversion_dir, "ref.txt", ".txt", temp_paths)
        hyp_txt = _conversion_path(conversion_dir, "hyp.txt", ".txt", temp_paths)
        ref_sidecar = _conversion_path(conversion_dir, "ref.sidecar.json", ".json", temp_paths)
        hyp_sidecar = _conversion_path(conversion_dir, "hyp.sidecar.json", ".json", temp_paths)
        ref_norm_stm = _conversion_path(conversion_dir, "ref.normalized.stm", ".stm", temp_paths)
        hyp_norm_stm = _conversion_path(conversion_dir, "hyp.normalized.stm", ".stm", temp_paths)
        conversion_trace = [
            convert_stm_to_txt(
                input_stm=ref_file,
                output_txt=ref_txt,
                sidecar_json=ref_sidecar,
                conversion_id=CONVERSION_ID,
            ),
            convert_stm_to_txt(
                input_stm=hyp_file,
                output_txt=hyp_txt,
                sidecar_json=hyp_sidecar,
                conversion_id=CONVERSION_ID,
            ),
        ]
        normalized_files, norm_result = _normalize_sa_asr_files(
            KeyTextFiles(ref_file=ref_txt, hyp_file=hyp_txt),
            language=language,
            normalization_node=resolved_normalization_node,
        )
        conversion_trace.extend(
            [
                convert_txt_to_stm(
                    input_txt=normalized_files.ref_file,
                    sidecar_json=ref_sidecar,
                    output_stm=ref_norm_stm,
                    conversion_id=CONVERSION_ID,
                ),
                convert_txt_to_stm(
                    input_txt=normalized_files.hyp_file,
                    sidecar_json=hyp_sidecar,
                    output_stm=hyp_norm_stm,
                    conversion_id=CONVERSION_ID,
                ),
            ]
        )
        _, scoring_result = score_meeteval(
            ref_file=ref_norm_stm,
            hyp_file=hyp_norm_stm,
            metric="cpwer",
            collar=collar,
            companion_metrics=companion_metrics,
        )
        trace = (norm_result, scoring_result)
        result = scoring_result.details["result"]
        return EvaluationReport(
            task="SA-ASR",
            language=language,
            metric="cpwer",
            score=float(result["cpwer"]),
            pipeline_id=pipeline_id,
            pipeline_trace=trace,
            input_contract=_SA_ASR_CONTRACT,
            input_files=input_files,
            computation_node_ids=component_trace_ids(components),
            details={
                "scoring_result": result,
                "conversion_trace": conversion_trace,
                "input_contract": _SA_ASR_CONTRACT.as_dict(),
                "input_files": input_files.as_dict(),
                "params": {
                    "collar": collar,
                    "companion_metrics": list(companion_metrics),
                    "normalization_node": resolved_normalization_node,
                },
            },
        )
    finally:
        _cleanup_trace_temp_files(trace)
        if conversion_dir is None:
            for path in temp_paths:
                Path(path).unlink(missing_ok=True)


def _resolve_normalization_node(*, language: str, normalization_node: str | None) -> str:
    normalized_language = language.lower().replace("_", "-")
    requested = (normalization_node or "").strip().lower()
    aliases = {
        "": "",
        "gstar": "normalization/gstar_norm",
        "gstar_norm": "normalization/gstar_norm",
        "normalization/gstar_norm": "normalization/gstar_norm",
        "whisper": "normalization/whisper_norm",
        "whisper_norm": "normalization/whisper_norm",
        "normalization/whisper_norm": "normalization/whisper_norm",
    }
    if requested not in aliases:
        raise ValueError(f"Unsupported SA-ASR normalization node: {normalization_node!r}")
    resolved = aliases[requested]
    if not resolved:
        if normalized_language in {"zh", "zh-cn", "cmn"}:
            return "normalization/gstar_norm"
        if normalized_language in {"en", "en-us", "en-gb"}:
            return "normalization/whisper_norm"
        raise ValueError(f"Unsupported SA-ASR language: {language!r}; supported: en, zh")
    if normalized_language in {"zh", "zh-cn", "cmn"} and resolved != "normalization/gstar_norm":
        raise ValueError("SA-ASR language='zh' requires normalization/gstar_norm")
    if normalized_language in {"en", "en-us", "en-gb"} and resolved != "normalization/whisper_norm":
        raise ValueError("SA-ASR language='en' requires normalization/whisper_norm")
    if normalized_language not in {"zh", "zh-cn", "cmn", "en", "en-us", "en-gb"}:
        raise ValueError(f"Unsupported SA-ASR language: {language!r}; supported: en, zh")
    return resolved


def _normalization_component(normalization_node: str):
    if normalization_node == "normalization/gstar_norm":
        return node_component("normalization/gstar_norm")
    if normalization_node == "normalization/whisper_norm":
        return node_component("normalization/whisper_norm", profile="english")
    raise ValueError(f"Unsupported SA-ASR normalization node: {normalization_node!r}")


def _normalize_sa_asr_files(
    files: KeyTextFiles,
    *,
    language: str,
    normalization_node: str,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    if normalization_node == "normalization/gstar_norm":
        return normalize_gstar_sa_asr_files(files, language=language)
    if normalization_node == "normalization/whisper_norm":
        return normalize_whisper_asr_files(files, language=language, profile="english")
    raise ValueError(f"Unsupported SA-ASR normalization node: {normalization_node!r}")


def _cleanup_trace_temp_files(trace: tuple[PipelineNodeResult, ...]) -> None:
    for result in trace:
        for key in ("ref_file", "hyp_file"):
            value = result.details.get(key)
            if isinstance(value, str):
                Path(value).unlink(missing_ok=True)


def _new_temp_path(suffix: str, temp_paths: list[str]) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    temp_paths.append(path)
    return path


def _conversion_path(
    conversion_dir: Path | None,
    filename: str,
    temp_suffix: str,
    temp_paths: list[str],
) -> str:
    if conversion_dir is not None:
        return str(conversion_dir / filename)
    return _new_temp_path(temp_suffix, temp_paths)
