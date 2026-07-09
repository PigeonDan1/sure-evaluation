"""KWS task-level evaluation routes."""

from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import KWSMetric, KWSSample
from sure_eval.evaluation.tasks.kws.compat import KWSMetricPipeline, KWSMetricReport, report_to_dict
from sure_eval.evaluation.tasks.kws.loaders import (
    load_samples_from_jsonl_and_outputs,
    load_samples_from_wekws_frame_score_file,
    load_samples_from_wekws_score_file,
)
from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_files, evaluate_kws_samples

__all__ = [
    "KWSMetric",
    "KWSMetricPipeline",
    "KWSMetricReport",
    "KWSSample",
    "evaluate_kws_files",
    "evaluate_kws_samples",
    "load_samples_from_jsonl_and_outputs",
    "load_samples_from_wekws_frame_score_file",
    "load_samples_from_wekws_score_file",
    "report_to_dict",
]
