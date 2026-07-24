"""SLU task route built from prompt normalization and classification scoring."""

from __future__ import annotations

from pathlib import Path

from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, KeyTextFiles, MetricInputContract
from sure_eval.evaluation.nodes.normalization.prompt_norm import normalize_prompt_choice_files
from sure_eval.evaluation.nodes.scoring.classify import default_label_spec, score_classification_files
from sure_eval.evaluation.pipeline_identity import build_atomic_pipeline_id, component_trace_ids, node_component

_SLU_CONTRACT = MetricInputContract(
    metric_id="normalization/prompt_norm+scoring/classify",
    required_roles=("hyp", "ref", "prompt_jsonl"),
    optional_roles=("label_spec",),
    row_format="key_label_with_prompt_choices",
    alignment_key="key",
    aggregation="accuracy",
    purpose="spoken_language_understanding_choice_accuracy",
)


def evaluate_slu_files(
    ref_file: str,
    hyp_file: str,
    *,
    prompt_jsonl: str,
    output_mode: str = "choice_id",
) -> EvaluationReport:
    """Evaluate SLU choices through prompt normalization then classify scoring."""

    input_files = EvaluationFiles(
        roles={
            "ref": ref_file,
            "hyp": hyp_file,
            "prompt_jsonl": prompt_jsonl,
        }
    )
    _SLU_CONTRACT.validate(input_files)
    normalized = None
    try:
        normalized, prompt_result = normalize_prompt_choice_files(
            KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file),
            prompt_jsonl=prompt_jsonl,
            output_mode=output_mode,
        )
        _, scoring_result = score_classification_files(
            ref_file=normalized.ref_file,
            hyp_file=normalized.hyp_file,
            label_spec=default_label_spec("SLU"),
            task="SLU",
        )
        result = scoring_result.details["result"]
        components = (
            node_component("normalization/prompt_norm", profile=output_mode),
            node_component("scoring/classify"),
        )
        pipeline_id = build_atomic_pipeline_id("slu", "any", "accuracy", components)
        return EvaluationReport(
            task="SLU",
            language="n/a",
            metric="accuracy",
            score=float(result["score"]),
            pipeline_id=pipeline_id,
            pipeline_trace=(prompt_result, scoring_result),
            input_contract=_SLU_CONTRACT,
            input_files=input_files,
            computation_node_ids=component_trace_ids(components),
            details={
                "scoring_result": result,
                "input_contract": _SLU_CONTRACT.as_dict(),
                "input_files": input_files.as_dict(),
                "normalized_files": {
                    "ref": normalized.ref_file,
                    "hyp": normalized.hyp_file,
                },
            },
        )
    finally:
        if normalized is not None:
            Path(normalized.ref_file).unlink(missing_ok=True)
            Path(normalized.hyp_file).unlink(missing_ok=True)
