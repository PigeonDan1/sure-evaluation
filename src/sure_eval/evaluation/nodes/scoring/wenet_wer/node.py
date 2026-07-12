"""WeNet-compatible ASR scoring wrappers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.scoring.wenet_wer.wenet_compute_cer import compute_wer

NODE_VERSION = "v1"
INTERNAL_STAGES = ("tokenization", "case_normalization", "edit_distance")
_MISSING_KEYS_SAMPLE_LIMIT = 10


def score_wenet_wer(files: KeyTextFiles) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score normalized key-text files with WeNet WER semantics."""

    result = _compute_wer_with_coverage(files.ref_file, files.hyp_file, tochar=False)
    _require_scored_tokens(result, metric="wer", ref_file=files.ref_file, hyp_file=files.hyp_file)
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

    result = _compute_wer_with_coverage(files.ref_file, files.hyp_file, tochar=True)
    _require_scored_tokens(result, metric="cer", ref_file=files.ref_file, hyp_file=files.hyp_file)
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
    mer_result = _compute_wer_with_coverage(side_outputs["ref_file"], side_outputs["hyp_file"])
    _require_scored_tokens(
        mer_result,
        metric="mer",
        ref_file=side_outputs["ref_file"],
        hyp_file=side_outputs["hyp_file"],
    )
    # zh-only / en-only side scores may legitimately cover zero tokens
    # (e.g. a monolingual subset), so they stay lenient.
    cer_result = _compute_wer_with_coverage(side_outputs["ref_zh_file"], side_outputs["hyp_zh_file"], tochar=True)
    wer_result = _compute_wer_with_coverage(side_outputs["ref_en_file"], side_outputs["hyp_en_file"])
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


def _compute_wer_with_coverage(ref_file: str, hyp_file: str, *, tochar: bool = False) -> dict:
    """Run ``compute_wer`` with utterance-coverage accounting.

    ``compute_wer`` silently skips reference utterances whose key is absent
    from the hypothesis file. This wrapper scores those utterances as empty
    hypotheses (pure deletions) instead, and reports coverage counts so a
    partial or mismatched hypothesis file can never pass as a clean run.
    """

    ref_keys = _read_utt_keys(ref_file)
    hyp_keys = _read_utt_keys(hyp_file)
    hyp_key_set = set(hyp_keys)
    ref_key_set = set(ref_keys)
    missing_keys = [key for key in ref_keys if key not in hyp_key_set]
    extra_keys = [key for key in hyp_keys if key not in ref_key_set]

    scored_hyp_file = hyp_file
    filled_hyp_file: str | None = None
    try:
        if missing_keys:
            filled_hyp_file = _fill_missing_hyp_utts(hyp_file, missing_keys)
            scored_hyp_file = filled_hyp_file
        result = compute_wer(ref_file, scored_hyp_file, tochar=tochar)
    finally:
        if filled_hyp_file is not None:
            Path(filled_hyp_file).unlink(missing_ok=True)

    result["num_ref_utts"] = len(ref_keys)
    result["num_hyp_utts"] = len(hyp_keys)
    result["num_hyp_missing_utts"] = len(missing_keys)
    result["num_hyp_extra_utts"] = len(extra_keys)
    if missing_keys:
        result["hyp_missing_keys_sample"] = missing_keys[:_MISSING_KEYS_SAMPLE_LIMIT]
        result["hyp_missing_policy"] = "scored_as_empty_hypothesis"
    if extra_keys:
        result["hyp_extra_keys_sample"] = extra_keys[:_MISSING_KEYS_SAMPLE_LIMIT]
    return result


def _read_utt_keys(path: str) -> list[str]:
    # Key extraction mirrors compute_wer: the first whitespace-delimited token.
    keys: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                keys.append(parts[0])
    return keys


def _fill_missing_hyp_utts(hyp_file: str, missing_keys: list[str]) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        with open(hyp_file, "r", encoding="utf-8") as source:
            content = source.read()
        handle.write(content)
        if content and not content.endswith("\n"):
            handle.write("\n")
        for key in missing_keys:
            handle.write(f"{key}\t\n")
    finally:
        handle.close()
    return handle.name


def _require_scored_tokens(result: dict, *, metric: str, ref_file: str, hyp_file: str) -> None:
    if result["all"] == 0:
        raise ValueError(
            f"ASR {metric} scoring covered zero reference tokens "
            f"({result.get('num_ref_utts', 0)} reference utterances parsed from {ref_file!r}, "
            f"{result.get('num_hyp_utts', 0)} hypothesis utterances parsed from {hyp_file!r}). "
            "Refusing to report a 0.0 error rate; check that both inputs are non-empty "
            "tab-separated <key>\\t<text> files with matching keys."
        )


def _rate(result: dict) -> float:
    if result["all"] == 0:
        return 0.0
    return (result["sub"] + result["del"] + result["ins"]) / result["all"]
