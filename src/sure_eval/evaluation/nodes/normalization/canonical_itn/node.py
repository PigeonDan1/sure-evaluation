"""Canonical written-form (ITN) normalization node for token-level CER."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.aispeech_norm.node import (
    _parse_key_text_line,
    _require_parsed_rows,
)
from sure_eval.evaluation.nodes.normalization.canonical_itn import chain

NODE_ID = "normalization/canonical_itn"
NODE_VERSION = "v1"
INTERNAL_STAGES = (
    "nfkc_lowercase",
    "idiom_unit_masking",
    "cn2an_itn",
    "cjk_numeral_span_pass",
    "percent_rewrite",
    "mixed_number_expansion",
    "punctuation_spacing",
)


def normalize_canonical_asr_files(
    files: KeyTextFiles, *, language: str
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize key-text files into the canonical written form."""

    if language not in {"zh", "en", "cs"}:
        raise ValueError(
            f"normalization/canonical_itn supports languages zh/en/cs, got {language!r}"
        )
    engine = chain.engine_info()  # fail fast when the ITN engine is missing
    # zh uses the plain chain; en/cs additionally whisper-normalize latin
    # spans (contractions, spoken numbers, fillers, spelling). Text without
    # latin letters is identical under both, which preserves the pure-Chinese
    # degeneration to cer_canonical.
    normalize_fn = chain.normalize_text if language == "zh" else chain.normalize_text_mixed
    if language != "zh":
        engine = dict(engine, en_span_normalizer="whisper_english")

    fallbacks_before = chain.itn_fallback_count()
    ref_norm_file = _new_temp_file()
    hyp_norm_file = _new_temp_file()
    try:
        ref_stats = _write_normalized_file(files.ref_file, ref_norm_file, normalize_fn)
        hyp_stats = _write_normalized_file(files.hyp_file, hyp_norm_file, normalize_fn)
    except Exception:
        Path(ref_norm_file).unlink(missing_ok=True)
        Path(hyp_norm_file).unlink(missing_ok=True)
        raise

    return (
        KeyTextFiles(ref_file=ref_norm_file, hyp_file=hyp_norm_file),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "language": language,
                "profile": f"{language}_canonical",
                "text_normalizer": "canonical_itn_chain",
                "ref_file": ref_norm_file,
                "hyp_file": hyp_norm_file,
                "row_stats": {"ref": ref_stats, "hyp": hyp_stats},
                "itn_fallback_rows": chain.itn_fallback_count() - fallbacks_before,
                "engine": engine,
            },
            internal_stages=INTERNAL_STAGES,
        ),
    )


def _write_normalized_file(input_file: str, output_file: str, normalize_fn) -> dict[str, int]:
    rows: list[str] = []
    num_empty_text_rows = 0
    num_dropped_malformed_rows = 0
    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            key, text = _parse_key_text_line(line)
            if key is None:
                if text is not None:
                    num_dropped_malformed_rows += 1
                continue
            if text:
                text_norm = normalize_fn(text)
            else:
                num_empty_text_rows += 1
                text_norm = ""
            rows.append(f"{key}\t{text_norm}")
    _require_parsed_rows(input_file, num_rows=len(rows), num_malformed=num_dropped_malformed_rows)
    Path(output_file).write_text("\n".join(rows) + "\n" if rows else "", encoding="utf-8")
    return {
        "num_rows": len(rows),
        "num_empty_text_rows": num_empty_text_rows,
        "num_dropped_malformed_rows": num_dropped_malformed_rows,
    }


def _new_temp_file() -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    return path
