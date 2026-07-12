"""Token-level CER scoring over canonical written-form text.

Tokens are produced by the canonical_itn tokenizer (CJK one token per char,
latin letter runs one token per word, digits one token per char, surviving
symbols one token each). The score is a corpus micro-average:
``(sub + del + ins) / total_reference_tokens`` with the S/D/I decomposition
taken from minimal edit operations (rapidfuzz, unit costs).

Utterance coverage follows the same policy as the other ASR scoring nodes:
reference utterances missing from the hypothesis file are scored as empty
hypotheses (pure deletions) and reported, hypothesis-only utterances are
ignored but counted, and a run covering zero reference tokens raises instead
of reporting a perfect 0.0.
"""

from __future__ import annotations

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.aispeech_norm.node import _parse_key_text_line
from sure_eval.evaluation.nodes.normalization.canonical_itn.chain import tokenize

NODE_ID = "scoring/token_cer"
NODE_VERSION = "v1"
INTERNAL_STAGES = ("canonical_tokenization", "token_edit_distance", "sdi_decomposition")
_MISSING_KEYS_SAMPLE_LIMIT = 10


def _require_rapidfuzz():
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "scoring/token_cer requires the 'rapidfuzz' package. "
            "Install it with: pip install \"sure-evaluation[canonical]\"."
        ) from exc
    return Levenshtein


def score_token_cer(files: KeyTextFiles) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score canonical key-text files with token-level edit distance."""

    levenshtein = _require_rapidfuzz()

    ref_rows = _read_rows(files.ref_file)
    hyp_rows = dict(_read_rows(files.hyp_file))
    ref_keys = [key for key, _ in ref_rows]
    ref_key_set = set(ref_keys)
    missing_keys = [key for key in ref_keys if key not in hyp_rows]
    extra_keys = [key for key in hyp_rows if key not in ref_key_set]

    total_ref = cor = sub = dele = ins = 0
    for key, ref_text in ref_rows:
        ref_tokens = tokenize(ref_text)
        hyp_tokens = tokenize(hyp_rows.get(key, ""))
        total_ref += len(ref_tokens)
        s = d = i = 0
        for op in levenshtein.editops(ref_tokens, hyp_tokens):
            if op.tag == "replace":
                s += 1
            elif op.tag == "delete":
                d += 1
            else:
                i += 1
        sub += s
        dele += d
        ins += i
        cor += len(ref_tokens) - s - d

    result: dict = {
        "all": total_ref,
        "cor": cor,
        "sub": sub,
        "del": dele,
        "ins": ins,
        "num_ref_utts": len(ref_rows),
        "num_hyp_utts": len(hyp_rows),
        "num_hyp_missing_utts": len(missing_keys),
        "num_hyp_extra_utts": len(extra_keys),
        "engine": {
            "distance": "rapidfuzz",
            "distance_version": _rapidfuzz_version(),
            "tokenization": "cjk_char/latin_word/digit_char",
        },
    }
    if missing_keys:
        result["hyp_missing_keys_sample"] = missing_keys[:_MISSING_KEYS_SAMPLE_LIMIT]
        result["hyp_missing_policy"] = "scored_as_empty_hypothesis"
    if extra_keys:
        result["hyp_extra_keys_sample"] = extra_keys[:_MISSING_KEYS_SAMPLE_LIMIT]

    if total_ref == 0:
        raise ValueError(
            "ASR cer_canonical scoring covered zero reference tokens "
            f"({len(ref_rows)} reference utterances parsed from {files.ref_file!r}, "
            f"{len(hyp_rows)} hypothesis utterances parsed from {files.hyp_file!r}). "
            "Refusing to report a 0.0 error rate; check that both inputs are non-empty "
            "tab-separated <key>\\t<text> files with matching keys."
        )

    score = (sub + dele + ins) / total_ref
    result["cer"] = score
    result["cer_percent"] = score * 100
    result["score"] = score
    return (
        files,
        PipelineNodeResult(
            stage="scoring",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={"metric": "cer_canonical", "result": result},
            internal_stages=INTERNAL_STAGES,
        ),
    )


def _read_rows(path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            key, text = _parse_key_text_line(line)
            if key is None:
                continue
            rows.append((key, text))
    return rows


def _rapidfuzz_version() -> str:
    import rapidfuzz

    return getattr(rapidfuzz, "__version__", "unknown")
