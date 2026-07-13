"""Punctuation-only text normalization for key-text evaluation files."""

from __future__ import annotations

import string
import tempfile
import unicodedata
from pathlib import Path

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult

NODE_ID = "normalization/punctuation_strip_norm"
NODE_VERSION = "v1"
PROFILE = "unicode_category_p_or_ascii"
ASCII_PUNCTUATION = frozenset(string.punctuation)


def normalize_punctuation_strip_text(text: str) -> str:
    """Remove punctuation characters without applying text normalization."""

    return "".join(ch for ch in text if not _is_punctuation(ch))


def normalize_punctuation_strip_key_text_files(
    files: KeyTextFiles,
    *,
    language: str = "auto",
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Strip punctuation from reference and hypothesis key-text files."""

    ref_file = _new_temp_file()
    hyp_file = _new_temp_file()
    try:
        ref_rows = _normalize_key_text_file(files.ref_file, ref_file)
        hyp_rows = _normalize_key_text_file(files.hyp_file, hyp_file)
    except Exception:
        Path(ref_file).unlink(missing_ok=True)
        Path(hyp_file).unlink(missing_ok=True)
        raise

    return (
        KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "language": language,
                "profile": PROFILE,
                "input_schema": "key_text_files",
                "output_schema": "key_text_files",
                "normalization": {
                    "operation": "punctuation_stripping_only",
                    "policy": "remove Unicode punctuation categories and ASCII punctuation",
                    "unicode_category_prefix": "P",
                    "ascii_punctuation": "".join(sorted(ASCII_PUNCTUATION)),
                    "preserves_numbers": True,
                    "preserves_case": True,
                    "preserves_whitespace": True,
                    "text_normalization": None,
                },
                "ref_file": ref_file,
                "hyp_file": hyp_file,
                "num_rows": {"ref": len(ref_rows), "hyp": len(hyp_rows)},
                "num_empty_after_normalization": {
                    "ref": sum(1 for row in ref_rows if not row["normalized_text"]),
                    "hyp": sum(1 for row in hyp_rows if not row["normalized_text"]),
                },
                "ref_rows": ref_rows,
                "hyp_rows": hyp_rows,
            },
            internal_stages=("key_text_parse", "punctuation_stripping", "key_text_write"),
        ),
    )


def _is_punctuation(ch: str) -> bool:
    return ch in ASCII_PUNCTUATION or unicodedata.category(ch).startswith("P")


def _normalize_key_text_file(input_file: str, output_file: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with (
        open(input_file, encoding="utf-8") as fin,
        open(output_file, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if "\t" not in line:
                continue
            key, original_text = line.rstrip("\n").split("\t", 1)
            normalized_text = normalize_punctuation_strip_text(original_text)
            fout.write(f"{key}\t{normalized_text}\n")
            rows.append(
                {
                    "key": key,
                    "original_text": original_text,
                    "normalized_text": normalized_text,
                }
            )
    return rows


def _new_temp_file() -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    return path
