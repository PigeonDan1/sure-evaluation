"""GigaSpeechBench-compatible ASR text normalization.

This node wraps the ``text_norm`` package published with GigaSpeechBench so
SURE-EVAL can reproduce the benchmark's reported WER/CER numbers.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Callable

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.giga_norm.normalization_impl import (
    PROFILE_ALIASES,
    SUPPORTED_PROFILES,
    get_normalizer,
    resolve_profile,
)

NODE_ID = "normalization/giga_norm"
NODE_VERSION = "v1"

UPSTREAM_PROJECT = "GigaSpeechBench"
UPSTREAM_URL = "https://github.com/SpeechColab/GigaSpeechBench"
UPSTREAM_COMMIT = "ca782bf"
UPSTREAM_SUBPATH = "text_norm"

# Dependencies whose versions change normalization output and therefore scores.
_SCORE_AFFECTING_PACKAGES = ("whisper_normalizer", "zhconv", "num2words", "underthesea", "regex")

# ASR route language -> GigaSpeechBench profile.
LANGUAGE_PROFILES = {
    "zh": "CHN",
    "en": "USA",
    "ja": "JPN",
    "ko": "KOR",
    "th": "THA",
    "id": "IDN",
    "ms": "MYS",
    "fil": "PHL",
    "vi": "VNM",
    "ar": "AR",
    "ar-ae": "ARE",
    "ar-dz": "DZA",
    "ar-eg": "EGY",
    "ar-iq": "IRQ",
    "ar-ma": "MAR",
    "ar-sa": "SAU",
    "ar-sy": "SYR",
}


def normalize_giga_asr_files(
    files: KeyTextFiles,
    *,
    language: str,
    profile: str | None = None,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize key-text ASR files with the GigaSpeechBench text rules."""

    resolved_profile = resolve_profile(profile or profile_for_language(language))
    normalizer = _cached_normalizer(resolved_profile)

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
                "profile": resolved_profile,
                "requested_profile": profile or profile_for_language(language),
                "input_schema": "key_text_files",
                "output_schema": "key_text_files",
                "ref_file": ref_file,
                "hyp_file": hyp_file,
                "num_rows": {"ref": len(ref_rows), "hyp": len(hyp_rows)},
                "num_empty_after_normalization": {
                    "ref": sum(1 for row in ref_rows if not row["normalized_text"]),
                    "hyp": sum(1 for row in hyp_rows if not row["normalized_text"]),
                },
                "normalization": giga_runtime_details(resolved_profile),
                "ref_rows": ref_rows,
                "hyp_rows": hyp_rows,
            },
            internal_stages=("key_text_parse", f"giga_{resolved_profile.lower()}_normalize"),
        ),
    )


def normalize_giga_text(text: str, *, profile: str) -> str:
    """Normalize one text string with the selected GigaSpeechBench profile."""

    return _cached_normalizer(resolve_profile(profile))(text).strip()


def profile_for_language(language: str) -> str:
    """Map an ASR route language tag to its GigaSpeechBench profile."""

    normalized = str(language).lower().strip().replace("_", "-")
    if normalized in LANGUAGE_PROFILES:
        return LANGUAGE_PROFILES[normalized]
    supported = ", ".join(sorted(LANGUAGE_PROFILES))
    raise ValueError(
        f"giga_norm is not mapped for ASR language={language!r}; supported languages: {supported}"
    )


def giga_runtime_details(profile: str) -> dict[str, object]:
    """Return trace-ready provenance for a resolved profile."""

    resolved = resolve_profile(profile)
    return {
        "backend": "gigaspeechbench_text_norm",
        "profile": resolved,
        "upstream_project": UPSTREAM_PROJECT,
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_subpath": UPSTREAM_SUBPATH,
        "vendored": True,
        "package_versions": {
            name: _package_version(name) for name in _SCORE_AFFECTING_PACKAGES
        },
    }


@lru_cache(maxsize=len(SUPPORTED_PROFILES))
def _cached_normalizer(profile: str) -> Callable[[str], str]:
    return get_normalizer(profile)


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


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


__all__ = [
    "LANGUAGE_PROFILES",
    "NODE_ID",
    "NODE_VERSION",
    "PROFILE_ALIASES",
    "SUPPORTED_PROFILES",
    "giga_runtime_details",
    "normalize_giga_asr_files",
    "normalize_giga_text",
    "profile_for_language",
]
