"""OpenAI Whisper-compatible text normalization wrappers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.whisper_norm.normalization_impl import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
)

NODE_ID = "normalization/whisper_norm"
NODE_VERSION = "v1"
UPSTREAM_PACKAGE = "openai-whisper"
UPSTREAM_VERSION = "20250625"
UPSTREAM_URL = "https://github.com/openai/whisper/tree/main/whisper/normalizers"


def normalize_whisper_asr_files(
    files: KeyTextFiles,
    *,
    language: str = "en",
    profile: str = "english",
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize key-text ASR files with OpenAI Whisper text rules."""

    normalizer = _normalizer_for_profile(profile)
    ref_file = _new_temp_file()
    hyp_file = _new_temp_file()
    try:
        ref_rows = _normalize_key_text_file(files.ref_file, ref_file, normalizer)
        hyp_rows = _normalize_key_text_file(files.hyp_file, hyp_file, normalizer)
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
                "profile": profile,
                "input_schema": "key_text_files",
                "output_schema": "key_text_files",
                "ref_file": ref_file,
                "hyp_file": hyp_file,
                "num_rows": {"ref": len(ref_rows), "hyp": len(hyp_rows)},
                "num_empty_after_normalization": {
                    "ref": sum(1 for row in ref_rows if not row["normalized_text"]),
                    "hyp": sum(1 for row in hyp_rows if not row["normalized_text"]),
                },
                "normalization": {
                    "backend": "openai_whisper",
                    "profile": profile,
                    "normalizer_class": _normalizer_class_name(profile),
                    "upstream_package": UPSTREAM_PACKAGE,
                    "upstream_version": UPSTREAM_VERSION,
                    "upstream_url": UPSTREAM_URL,
                    "vendored": True,
                },
                "ref_rows": ref_rows,
                "hyp_rows": hyp_rows,
            },
            internal_stages=("key_text_parse", f"whisper_{profile}_normalize"),
        ),
    )


def normalize_whisper_text(text: str, *, profile: str = "english") -> str:
    """Normalize one text string with the selected Whisper profile."""

    return _normalizer_for_profile(profile)(text).strip()


def _normalizer_for_profile(profile: str) -> Callable[[str], str]:
    normalized = profile.lower()
    if normalized == "english":
        return EnglishTextNormalizer()
    if normalized == "basic":
        return BasicTextNormalizer()
    raise ValueError(f"Unsupported whisper_norm profile: {profile}")


def _normalizer_class_name(profile: str) -> str:
    normalized = profile.lower()
    if normalized == "english":
        return "EnglishTextNormalizer"
    if normalized == "basic":
        return "BasicTextNormalizer"
    raise ValueError(f"Unsupported whisper_norm profile: {profile}")


def _normalize_key_text_file(
    input_file: str,
    output_file: str,
    normalizer: Callable[[str], str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if "\t" not in line:
                continue
            key, original_text = line.rstrip("\n").split("\t", 1)
            normalized_text = normalizer(original_text).strip()
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
