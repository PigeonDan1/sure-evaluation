"""WeNet-compatible ASR scoring wrappers."""

from __future__ import annotations

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.scoring.wenet_wer.wenet_compute_cer import compute_wer

NODE_VERSION = "v1"
INTERNAL_STAGES = ("tokenization", "case_normalization", "edit_distance")


def score_wenet_wer(files: KeyTextFiles) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score normalized key-text files with WeNet WER semantics."""

    result = compute_wer(files.ref_file, files.hyp_file, tochar=False)
    score = _rate(result)
    result["wer"] = score
    result["wer_percent"] = score * 100
    result["score"] = score
    return (
        files,
        PipelineNodeResult(
            stage="scoring",
            node_id="scoring/wenet_wer",
            version=NODE_VERSION,
            details={"metric": "wer", "result": result},
            internal_stages=INTERNAL_STAGES,
        ),
    )


def score_wenet_cer(files: KeyTextFiles) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score normalized key-text files with WeNet CER semantics."""

    result = compute_wer(files.ref_file, files.hyp_file, tochar=True)
    score = _rate(result)
    result["cer"] = score
    result["cer_percent"] = score * 100
    result["score"] = score
    return (
        files,
        PipelineNodeResult(
            stage="scoring",
            node_id="scoring/wenet_cer",
            version=NODE_VERSION,
            details={"metric": "cer", "result": result},
            internal_stages=INTERNAL_STAGES,
        ),
    )


def score_codeswitch_mer(files: KeyTextFiles, *, normalization_details: dict) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score code-switch ASR with the same MER/WER/CER decomposition as legacy ASR."""

    side_outputs = normalization_details["side_outputs"]
    mer_result = compute_wer(side_outputs["ref_file"], side_outputs["hyp_file"])
    cer_result = compute_wer(side_outputs["ref_zh_file"], side_outputs["hyp_zh_file"], tochar=True)
    wer_result = compute_wer(side_outputs["ref_en_file"], side_outputs["hyp_en_file"])
    mer_score = _rate(mer_result)
    cer_score = _rate(cer_result)
    wer_score = _rate(wer_result)
    result = {
        "mer": mer_score,
        "wer": wer_score,
        "cer": cer_score,
        "mer_percent": mer_score * 100,
        "wer_percent": wer_score * 100,
        "cer_percent": cer_score * 100,
        "score": mer_score,
        "mer_details": mer_result,
        "wer_details": wer_result,
        "cer_details": cer_result,
    }
    return (
        files,
        PipelineNodeResult(
            stage="scoring",
            node_id="scoring/wenet_mer",
            version=NODE_VERSION,
            details={"metric": "mer", "result": result},
            internal_stages=("mixed_token_scoring", "zh_cer_scoring", "en_wer_scoring"),
        ),
    )


def _rate(result: dict) -> float:
    if result["all"] == 0:
        return 0.0
    return (result["sub"] + result["del"] + result["ins"]) / result["all"]
