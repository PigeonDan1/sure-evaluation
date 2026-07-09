"""S2TT task routes built from reusable pipeline nodes."""

from __future__ import annotations

from sure_eval.evaluation.core.types import (
    EvaluationFiles,
    EvaluationReport,
    KeyTextFiles,
    MetricInputContract,
)
from sure_eval.evaluation.nodes.scoring.bleurt_20 import BLEURTRunner, score_bleurt_20
from sure_eval.evaluation.nodes.scoring.sacrebleu import score_sacrebleu
from sure_eval.evaluation.nodes.scoring.xcomet_xl import XCOMETRunner, score_xcomet_xl

_SACREBLEU_CONTRACT = MetricInputContract(
    metric_id="scoring/sacrebleu",
    required_roles=("hyp", "ref"),
    aggregation="corpus_metric",
    purpose="traditional_reproducible_anchor",
)
_XCOMET_XL_CONTRACT = MetricInputContract(
    metric_id="scoring/xcomet_xl",
    required_roles=("src", "hyp", "ref"),
    aggregation="segment_mean",
    purpose="primary_semantic_quality",
    model="Unbabel/XCOMET-XL",
)
_BLEURT_20_CONTRACT = MetricInputContract(
    metric_id="scoring/bleurt_20",
    required_roles=("hyp", "ref"),
    aggregation="segment_mean",
    purpose="complementary_semantic_quality",
    model="BLEURT-20",
)


def evaluate_s2tt_files(
    ref_file: str,
    hyp_file: str,
    *,
    language: str,
    metric: str = "bleu",
    src_file: str | None = None,
    xcomet_runner: XCOMETRunner | None = None,
    bleurt_runner: BLEURTRunner | None = None,
) -> EvaluationReport:
    """Evaluate S2TT key-text files through the configured task pipeline."""

    normalized_metric = metric.lower()
    if normalized_metric not in {"bleu", "bleu_char", "chrf", "xcomet_xl", "bleurt_20"}:
        raise ValueError(f"Unsupported S2TT metric: {metric}")

    key_text_files = KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file)
    if normalized_metric == "xcomet_xl":
        if src_file is None:
            raise ValueError("src_file is required for metric xcomet_xl")
        input_files = EvaluationFiles.from_src_ref_hyp(
            src_file=src_file,
            ref_file=ref_file,
            hyp_file=hyp_file,
        )
        input_contract = _XCOMET_XL_CONTRACT
        input_contract.validate(input_files)
        scoring_result = score_xcomet_xl(
            src_file=src_file,
            ref_file=ref_file,
            hyp_file=hyp_file,
            language=language,
            runner=xcomet_runner,
        )
        pipeline_profile = language
        pipeline_suffix = "xcomet_xl"
    elif normalized_metric == "bleurt_20":
        input_files = EvaluationFiles.from_ref_hyp(ref_file=ref_file, hyp_file=hyp_file)
        input_contract = _BLEURT_20_CONTRACT
        input_contract.validate(input_files)
        _, scoring_result = score_bleurt_20(
            key_text_files,
            language=language,
            runner=bleurt_runner,
        )
        pipeline_profile = language
        pipeline_suffix = "bleurt_20"
    else:
        input_files = EvaluationFiles.from_ref_hyp(ref_file=ref_file, hyp_file=hyp_file)
        input_contract = _SACREBLEU_CONTRACT
        input_contract.validate(input_files)
        _, scoring_result = score_sacrebleu(
            key_text_files,
            language=language,
        )
        pipeline_profile = scoring_result.details["tokenizer_profile"]
        pipeline_suffix = "sacrebleu"

    trace = (scoring_result,)
    result = scoring_result.details["result"]
    result_key = normalized_metric if normalized_metric != "bleu_char" else "bleu_char"
    return EvaluationReport(
        task="S2TT",
        language=language,
        metric=normalized_metric,
        score=float(result[result_key]),
        pipeline_id=f"s2tt.{pipeline_profile}.{normalized_metric}.{pipeline_suffix}",
        pipeline_trace=trace,
        input_contract=input_contract,
        input_files=input_files,
        details={
            "scoring_result": result,
            "input_contract": input_contract.as_dict(),
            "input_files": input_files.as_dict(),
        },
    )
