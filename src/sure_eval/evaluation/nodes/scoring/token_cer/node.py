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

import unicodedata

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

    return score_key_text_tokens(
        files,
        metric="cer_canonical",
        score_key="cer",
        node_id=NODE_ID,
        version=NODE_VERSION,
        internal_stages=INTERNAL_STAGES,
    )


def score_key_text_tokens(
    files: KeyTextFiles,
    *,
    metric: str,
    score_key: str,
    node_id: str,
    version: str,
    internal_stages: tuple[str, ...],
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Shared token-level scorer for the canonical metric family."""

    levenshtein = _require_rapidfuzz()

    ref_rows = _read_rows(files.ref_file)
    hyp_rows = dict(_read_rows(files.hyp_file))
    ref_keys = [key for key, _ in ref_rows]
    ref_key_set = set(ref_keys)
    missing_keys = [key for key in ref_keys if key not in hyp_rows]
    extra_keys = [key for key in hyp_rows if key not in ref_key_set]

    total_ref = cor = sub = dele = ins = 0
    spacing_repairs = 0
    for key, ref_text in ref_rows:
        ref_tokens = tokenize(ref_text)
        hyp_tokens = tokenize(hyp_rows.get(key, ""))
        repaired = _repair_word_spacing(ref_tokens, hyp_tokens)
        if repaired is not None:
            ref_tokens, hyp_tokens, n_repairs = repaired
            spacing_repairs += n_repairs
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
        "spacing_repairs": spacing_repairs,
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
            f"ASR {metric} scoring covered zero reference tokens "
            f"({len(ref_rows)} reference utterances parsed from {files.ref_file!r}, "
            f"{len(hyp_rows)} hypothesis utterances parsed from {files.hyp_file!r}). "
            "Refusing to report a 0.0 error rate; check that both inputs are non-empty "
            "tab-separated <key>\\t<text> files with matching keys."
        )

    score = (sub + dele + ins) / total_ref
    result[score_key] = score
    result[f"{score_key}_percent"] = score * 100
    result["score"] = score
    return (
        files,
        PipelineNodeResult(
            stage="scoring",
            node_id=node_id,
            version=version,
            details={"metric": metric, "result": result},
            internal_stages=internal_stages,
        ),
    )


_MAX_SPACING_PARTS = 4


def _repair_word_spacing(
    ref_tokens: list[str], hyp_tokens: list[str]
) -> tuple[list[str], list[str], int] | None:
    """Cancel pure spacing artifacts between latin word tokens.

    A word token on one side is split when its letters EXACTLY equal the
    concatenation of 2..4 consecutive word tokens on the other side
    ("tenthe" vs "ten the", "some thing" vs "something"). This forgives only
    the case where the recognized content is identical and word spacing is
    not — any letter difference leaves tokens untouched and fully scored.
    Deterministic, per-utterance, no external state. CJK / digit / symbol
    tokens never participate, so pure-CJK degeneration is unaffected.
    """

    def word_indices(tokens: list[str]) -> list[int]:
        return [
            i
            for i, tok in enumerate(tokens)
            if tok.isalpha() and unicodedata.category(tok[0]) != "Lo"
        ]

    def concat_map(tokens: list[str]) -> dict[str, tuple[str, ...]]:
        indices = word_indices(tokens)
        table: dict[str, tuple[str, ...]] = {}
        for start in range(len(indices)):
            for parts in range(2, _MAX_SPACING_PARTS + 1):
                end = start + parts
                if end > len(indices):
                    break
                idx = indices[start:end]
                # parts must be adjacent in the original token stream
                if idx[-1] - idx[0] != parts - 1:
                    break
                joined = "".join(tokens[i] for i in idx)
                table.setdefault(joined, tuple(tokens[i] for i in idx))
        return table

    def repair(tokens: list[str], other_concats: dict[str, tuple[str, ...]], other_words: set[str]):
        out: list[str] = []
        repairs = 0
        for tok in tokens:
            split = other_concats.get(tok)
            if (
                split is not None
                and len(tok) >= 4
                and tok.isalpha()
                and unicodedata.category(tok[0]) != "Lo"
                and tok not in other_words  # identical whole word: direct match, keep
            ):
                out.extend(split)
                repairs += 1
            else:
                out.append(tok)
        return out, repairs

    ref_concats = concat_map(ref_tokens)
    hyp_concats = concat_map(hyp_tokens)
    if not ref_concats and not hyp_concats:
        return None
    new_hyp, hyp_repairs = repair(hyp_tokens, ref_concats, set(ref_tokens))
    new_ref, ref_repairs = repair(ref_tokens, hyp_concats, set(hyp_tokens))
    if hyp_repairs == 0 and ref_repairs == 0:
        return None
    return new_ref, new_hyp, hyp_repairs + ref_repairs


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
