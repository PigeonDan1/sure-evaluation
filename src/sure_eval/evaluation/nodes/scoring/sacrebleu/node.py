"""SacreBLEU scoring wrapper for S2TT."""

from __future__ import annotations

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult

NODE_ID = "scoring/sacrebleu"
NODE_VERSION = "v1"


def score_sacrebleu(
    files: KeyTextFiles,
    *,
    language: str,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score key-text S2TT files with legacy SacreBLEU settings."""

    from sacrebleu.metrics import BLEU, CHRF

    tokenizer_profile, tokenizer = _tokenizer_for_language(language)
    ref_lines = _load_text_column(files.ref_file)
    hyp_lines = _load_text_column(files.hyp_file)
    bleu = BLEU(tokenize=tokenizer)
    chrf = CHRF(word_order=2)
    score_bleu = bleu.corpus_score(hyp_lines, [ref_lines])
    score_chrf = chrf.corpus_score(hyp_lines, [ref_lines])
    result = {
        "bleu": score_bleu.score,
        "bleu_char": score_bleu.score,
        "chrf": score_chrf.score,
        "score": score_bleu.score,
    }
    return (
        files,
        PipelineNodeResult(
            stage="scoring",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "backend": "sacrebleu",
                "metric": "bleu",
                "language": language,
                "tokenizer_profile": tokenizer_profile,
                "tokenizer": tokenizer,
                "result": result,
                "num_samples": len(ref_lines),
            },
            internal_stages=("tokenizer_selection", "corpus_bleu", "corpus_chrf2"),
        ),
    )


def _tokenizer_for_language(language: str) -> tuple[str, str]:
    normalized_language = language.lower()
    if normalized_language in {"zh", "ch", "chinese"}:
        return "zh", "zh"
    if normalized_language in {"en", "english"}:
        return "en", "13a"
    return "none", "none"


def _load_text_column(path: str) -> list[str]:
    rows: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                rows.append(parts[1])
    return rows
