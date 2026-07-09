"""Generic MeetEval scoring wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult

NODE_ID = "scoring/meeteval"
NODE_VERSION = "v1"

SUPPORTED_INPUT_FORMATS = ("STM", "CTM", "SegLST", "RTTM for DER")


def score_meeteval(
    *,
    ref_file: str,
    hyp_file: str,
    metric: str,
    collar: float | None = None,
    companion_metrics: tuple[str, ...] = (),
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score annotation files through MeetEval without constraining the input format."""

    normalized_metric = _normalize_metric(metric)
    normalized_companions = tuple(_normalize_metric(item) for item in companion_metrics)
    try:
        import meeteval
    except ImportError as exc:
        raise RuntimeError(
            "meeteval is required for SD and SA-ASR scoring. "
            "Install the node-local environment in src/sure_eval/evaluation/nodes/scoring/meeteval."
        ) from exc

    _ensure_md_eval_on_path()
    ref = meeteval.io.load(ref_file)
    hyp = meeteval.io.load(hyp_file)
    result = _score_loaded(
        meeteval_module=meeteval,
        reference=ref,
        hypothesis=hyp,
        metric=normalized_metric,
        collar=collar,
    )
    companion_results = {
        companion: _score_loaded(
            meeteval_module=meeteval,
            reference=ref,
            hypothesis=hyp,
            metric=companion,
            collar=collar,
        )
        for companion in normalized_companions
    }

    combined_result = dict(result)
    for companion, companion_result in companion_results.items():
        combined_result[companion] = companion_result[companion]
        if companion == "der":
            combined_result["num_sessions"] = companion_result.get("num_sessions", combined_result.get("num_sessions", 0))
    if "score" not in combined_result:
        combined_result["score"] = combined_result[normalized_metric]

    return (
        KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file),
        PipelineNodeResult(
            stage="scoring",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "backend": "meeteval",
                "metric": normalized_metric,
                "input_loader": "meeteval.io.load",
                "input_formats_supported": list(SUPPORTED_INPUT_FORMATS),
                "params": {
                    "collar": collar,
                    "companion_metrics": list(normalized_companions),
                },
                "aggregation": _aggregation_for_metric(normalized_metric),
                "result": combined_result,
            },
            internal_stages=("meeteval_load", normalized_metric, "result_aggregation"),
        ),
    )


def _score_loaded(
    *,
    meeteval_module: Any,
    reference: Any,
    hypothesis: Any,
    metric: str,
    collar: float | None,
) -> dict[str, Any]:
    if metric == "der":
        return _score_der(meeteval_module, reference, hypothesis, collar=0.25 if collar is None else collar)
    if metric == "cpwer":
        return _score_cpwer(meeteval_module, reference, hypothesis)
    raise ValueError(f"Unsupported MeetEval metric: {metric}")


def _score_der(meeteval_module: Any, reference: Any, hypothesis: Any, *, collar: float) -> dict[str, Any]:
    _ensure_md_eval_on_path()
    der_by_session = meeteval_module.der.dscore(reference, hypothesis, collar=collar)
    per_session = {
        str(session): _der_error_rate_as_dict(error_rate)
        for session, error_rate in der_by_session.items()
    }
    values = [session["error_rate"] for session in per_session.values()]
    der = sum(values) / len(values) if values else 0.0
    return {
        "metric_name": "der",
        "score": der,
        "der": der,
        "num_sessions": len(values),
        "aggregation": "session_mean_error_rate",
        "per_session": per_session,
    }


def _score_cpwer(meeteval_module: Any, reference: Any, hypothesis: Any) -> dict[str, Any]:
    cpwer_by_session = meeteval_module.wer.cpwer(reference, hypothesis)
    combined = meeteval_module.wer.combine_error_rates(cpwer_by_session.values())
    per_session = {
        str(session): _wer_error_rate_as_dict(error_rate)
        for session, error_rate in cpwer_by_session.items()
    }
    cpwer = float(combined.error_rate)
    return {
        "metric_name": "cpwer",
        "score": cpwer,
        "cpwer": cpwer,
        "num_sessions": len(per_session),
        "aggregation": "meeteval_combined_error_rate",
        "errors": _float_or_none(getattr(combined, "errors", None)),
        "length": _float_or_none(getattr(combined, "length", None)),
        "per_session": per_session,
    }


def _der_error_rate_as_dict(error_rate: Any) -> dict[str, Any]:
    return {
        "error_rate": float(error_rate.error_rate),
        "missed_speaker_time": _float_or_none(getattr(error_rate, "missed_speaker_time", None)),
        "falarm_speaker_time": _float_or_none(getattr(error_rate, "falarm_speaker_time", None)),
        "speaker_error_time": _float_or_none(getattr(error_rate, "speaker_error_time", None)),
    }


def _wer_error_rate_as_dict(error_rate: Any) -> dict[str, Any]:
    return {
        "error_rate": float(error_rate.error_rate),
        "errors": _float_or_none(getattr(error_rate, "errors", None)),
        "length": _float_or_none(getattr(error_rate, "length", None)),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalize_metric(metric: str) -> str:
    normalized = metric.strip().lower().replace("-", "_")
    aliases = {
        "der_dscore": "der",
        "dscore": "der",
        "cp_wer": "cpwer",
    }
    return aliases.get(normalized, normalized)


def _aggregation_for_metric(metric: str) -> str:
    if metric == "der":
        return "session_mean_error_rate"
    if metric == "cpwer":
        return "meeteval_combined_error_rate"
    return "meeteval_metric"


def _ensure_md_eval_on_path() -> None:
    """Expose a node-local md-eval-22.pl to MeetEval when available."""

    node_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    env_path = os.environ.get("SURE_EVAL_MD_EVAL_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            node_dir / "md-eval-22.pl",
            node_dir / ".cache" / "md-eval-22.pl",
        ]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        executable = candidate
        if not os.access(executable, os.X_OK):
            executable.chmod(executable.stat().st_mode | 0o755)
        path_dir = str(executable.parent)
        current_path = os.environ.get("PATH", "")
        if path_dir not in current_path.split(os.pathsep):
            os.environ["PATH"] = path_dir + os.pathsep + current_path
        return
