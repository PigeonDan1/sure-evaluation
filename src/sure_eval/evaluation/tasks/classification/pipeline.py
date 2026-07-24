"""Classification task route built on the generic classify scoring node."""

from __future__ import annotations

from pathlib import Path

from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, MetricInputContract
from sure_eval.evaluation.nodes.scoring.classify import LabelSpec, load_label_spec, score_classification_files
from sure_eval.evaluation.pipeline_identity import build_atomic_pipeline_id, component_trace_ids, node_component

_CLASSIFICATION_CONTRACT = MetricInputContract(
    metric_id="scoring/classify",
    required_roles=("hyp", "ref", "label_spec"),
    row_format="key_label",
    alignment_key="key",
    aggregation="accuracy",
    purpose="classification_accuracy",
)


def evaluate_classification_files(
    ref_file: str,
    hyp_file: str,
    *,
    task: str = "classification",
    label_spec: LabelSpec | str | Path | dict | None = None,
) -> EvaluationReport:
    """Evaluate aligned classification labels with a dataset label spec."""

    spec = load_label_spec(label_spec, task=task)
    input_files = EvaluationFiles(
        roles={
            "ref": ref_file,
            "hyp": hyp_file,
            "label_spec": spec.id if not isinstance(label_spec, (str, Path)) else str(label_spec),
        }
    )
    _CLASSIFICATION_CONTRACT.validate(input_files)
    _, scoring_result = score_classification_files(
        ref_file=ref_file,
        hyp_file=hyp_file,
        label_spec=spec,
        task=task,
    )
    result = scoring_result.details["result"]
    normalized_task = task.upper() if task.upper() in {"SER", "GR"} else task
    task_alias = normalized_task.lower()
    components = (node_component("scoring/classify"),)
    pipeline_id = build_atomic_pipeline_id(task_alias, "any", "accuracy", components)
    return EvaluationReport(
        task=normalized_task,
        language="n/a",
        metric="accuracy",
        score=float(result["score"]),
        pipeline_id=pipeline_id,
        pipeline_trace=(scoring_result,),
        input_contract=_CLASSIFICATION_CONTRACT,
        input_files=input_files,
        computation_node_ids=component_trace_ids(components),
        details={
            "scoring_result": result,
            "input_contract": _CLASSIFICATION_CONTRACT.as_dict(),
            "input_files": input_files.as_dict(),
            "label_spec": spec.as_dict(),
        },
    )
