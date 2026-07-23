"""KWS task routes built from reusable pipeline nodes."""

from __future__ import annotations

from pathlib import Path

from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, MetricInputContract
from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import KWSSample
from sure_eval.evaluation.nodes.scoring.wekws_det import score_wekws_det

_WEKWS_DET_CONTRACT = MetricInputContract(
    metric_id="scoring/wekws_det",
    required_roles=("samples",),
    row_format="kws_samples",
    alignment_key="key",
    aggregation="threshold_operating_point",
    purpose="keyword_detection_quality",
)

_SURE_JSON_CONTRACT = MetricInputContract(
    metric_id="scoring/wekws_det",
    required_roles=("reference_jsonl", "sample_output"),
    row_format="sure_kws_json",
    alignment_key="key",
    aggregation="threshold_operating_point",
    purpose="keyword_detection_quality",
)
_WEKWS_SCORE_CTC_CONTRACT = MetricInputContract(
    metric_id="scoring/wekws_det",
    required_roles=("wekws_label_file", "wekws_score_file", "keyword"),
    row_format="wekws_score_ctc",
    alignment_key="key",
    aggregation="threshold_operating_point",
    purpose="keyword_detection_quality",
)
_WEKWS_FRAME_SCORE_CONTRACT = MetricInputContract(
    metric_id="scoring/wekws_det",
    required_roles=("wekws_label_file", "wekws_frame_score_file", "keyword"),
    row_format="wekws_frame_score",
    alignment_key="key",
    aggregation="threshold_operating_point",
    purpose="keyword_detection_quality",
)

_SUPPORTED_PRIMARY_METRICS = {"accuracy", "macro-recall"}


def evaluate_kws_samples(
    samples: list[KWSSample],
    *,
    threshold: float = 0.5,
    thresholds: list[float] | None = None,
    threshold_step: float = 0.01,
    macro_recall_false_alarms: int = 0,
    profile: str = "default",
    metric: str = "accuracy",
    input_mode: str = "samples",
    input_contract: MetricInputContract | None = None,
    input_files: EvaluationFiles | None = None,
) -> EvaluationReport:
    """Evaluate aligned KWS samples through the configured task pipeline."""

    metric = metric.lower()
    if metric not in _SUPPORTED_PRIMARY_METRICS:
        raise ValueError(f"Unsupported KWS primary metric: {metric}")
    input_contract = input_contract or _WEKWS_DET_CONTRACT
    if input_files is not None:
        input_contract.validate(input_files)
    scoring_result = score_wekws_det(
        samples,
        threshold=threshold,
        thresholds=thresholds,
        threshold_step=threshold_step,
        macro_recall_false_alarms=macro_recall_false_alarms,
    )
    results = scoring_result.details["results"]
    return EvaluationReport(
        task="KWS",
        language="n/a",
        metric=metric,
        score=float(results[metric]["score"]),
        pipeline_id=f"kws.{profile}.{metric}.wekws_det",
        pipeline_trace=(scoring_result,),
        input_contract=input_contract,
        input_files=input_files,
        details={
            "scoring_result": scoring_result.details,
            "results": results,
            "rows": scoring_result.details["rows"],
            "summary": scoring_result.details["summary"],
            "input_mode": input_mode,
            "input_contract": input_contract.as_dict(),
            "input_files": input_files.as_dict() if input_files else {},
        },
    )


def evaluate_kws_files(
    *,
    reference_jsonl: str | Path | None = None,
    sample_output: str | Path | None = None,
    wekws_label_file: str | Path | None = None,
    wekws_score_file: str | Path | None = None,
    wekws_frame_score_file: str | Path | None = None,
    keyword: str | None = None,
    threshold: float = 0.5,
    thresholds: list[float] | None = None,
    threshold_step: float = 0.01,
    metric: str = "accuracy",
    macro_recall_false_alarms: int = 0,
) -> EvaluationReport:
    """Load supported KWS input files and evaluate them through the task route."""

    from sure_eval.evaluation.tasks.kws.loaders import (
        load_samples_from_jsonl_and_outputs,
        load_samples_from_wekws_frame_score_file,
        load_samples_from_wekws_score_file,
    )

    if reference_jsonl and sample_output:
        samples = load_samples_from_jsonl_and_outputs(reference_jsonl, sample_output)
        input_mode = "sure_json"
        input_contract = _SURE_JSON_CONTRACT
        input_files = EvaluationFiles(
            roles={
                "reference_jsonl": str(reference_jsonl),
                "sample_output": str(sample_output),
            }
        )
    elif wekws_label_file and wekws_score_file and keyword:
        samples = load_samples_from_wekws_score_file(
            wekws_label_file,
            wekws_score_file,
            keyword=keyword,
        )
        input_mode = "wekws_score_ctc"
        input_contract = _WEKWS_SCORE_CTC_CONTRACT
        input_files = EvaluationFiles(
            roles={
                "wekws_label_file": str(wekws_label_file),
                "wekws_score_file": str(wekws_score_file),
                "keyword": keyword,
            }
        )
    elif wekws_label_file and wekws_frame_score_file and keyword:
        samples = load_samples_from_wekws_frame_score_file(
            wekws_label_file,
            wekws_frame_score_file,
            keyword=keyword,
            threshold=threshold,
        )
        input_mode = "wekws_frame_score"
        input_contract = _WEKWS_FRAME_SCORE_CONTRACT
        input_files = EvaluationFiles(
            roles={
                "wekws_label_file": str(wekws_label_file),
                "wekws_frame_score_file": str(wekws_frame_score_file),
                "keyword": keyword,
            }
        )
    else:
        raise ValueError(
            "Provide either reference_jsonl + sample_output, "
            "wekws_label_file + wekws_score_file + keyword, or "
            "wekws_label_file + wekws_frame_score_file + keyword."
        )
    return evaluate_kws_samples(
        samples,
        threshold=threshold,
        thresholds=thresholds,
        threshold_step=threshold_step,
        macro_recall_false_alarms=macro_recall_false_alarms,
        profile=input_mode,
        metric=metric,
        input_mode=input_mode,
        input_contract=input_contract,
        input_files=input_files,
    )
