"""SLU task route built from prompt normalization and classification scoring."""

from __future__ import annotations

from pathlib import Path

from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, KeyTextFiles, MetricInputContract
from sure_eval.evaluation.nodes.normalization.prompt_norm import normalize_prompt_choice_files
from sure_eval.evaluation.nodes.scoring.classify import default_label_spec, score_classification_files
from sure_eval.evaluation.nodes.scoring.llm_judge.node import JudgeConfig, score_slu_judge
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
    ref_file: str | None = None,
    hyp_file: str | None = None,
    *,
    prompt_jsonl: str,
    output_mode: str = "choice_id",
    metric: str = "accuracy",
    llm_config: str | None = None,
) -> EvaluationReport:
    """Evaluate SLU answers through the selected scoring route."""

    if metric not in {"accuracy", "llm_judge"}:
        raise ValueError(f"Unsupported SLU metric: {metric}")
    contract = _SLU_CONTRACT
    if metric == "llm_judge":
        contract = MetricInputContract(
            metric_id="scoring/llm_judge", required_roles=("hyp", "prompt_jsonl"),
            optional_roles=("ref",), row_format="key_text_with_prompt_context",
            aggregation="segment_mean_complete_batch",
            purpose="textual_spoken_language_understanding_answer_quality",
        )

    input_files = EvaluationFiles(
        roles={
            "ref": ref_file,
            "hyp": hyp_file,
            "prompt_jsonl": prompt_jsonl,
        }
    )
    if ref_file is None:
        input_files.roles.pop("ref")
    contract.validate(input_files)
    normalized = None
    try:
        if metric == "llm_judge":
            if not llm_config:
                raise ValueError("SLU llm_judge requires llm_config")
            config = JudgeConfig.from_file(llm_config)
            scoring_result = score_slu_judge(
                hyp_file=hyp_file, prompt_jsonl=prompt_jsonl, ref_file=ref_file, config=config,
            )
            components = (node_component("scoring/llm_judge"),)
            trace = (scoring_result,)
        else:
            if llm_config is not None:
                raise ValueError("llm_config is only accepted by the llm_judge route")
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
            components = (
                node_component("normalization/prompt_norm", profile=output_mode),
                node_component("scoring/classify"),
            )
            trace = (prompt_result, scoring_result)
        result = scoring_result.details["result"]
        details = {
            "scoring_result": result,
            "input_contract": contract.as_dict(),
            "input_files": input_files.as_dict(),
        }
        if normalized is not None:
            details["normalized_files"] = {"ref": normalized.ref_file, "hyp": normalized.hyp_file}
        pipeline_id = build_atomic_pipeline_id("slu", "any", metric, components)
        return EvaluationReport(
            task="SLU",
            language="n/a",
            metric=metric,
            score=float(result["score"]) if result["score"] is not None else None,
            pipeline_id=pipeline_id,
            pipeline_trace=trace,
            input_contract=contract,
            input_files=input_files,
            computation_node_ids=component_trace_ids(components),
            details=details,
        )
    finally:
        if normalized is not None:
            Path(normalized.ref_file).unlink(missing_ok=True)
            Path(normalized.hyp_file).unlink(missing_ok=True)
