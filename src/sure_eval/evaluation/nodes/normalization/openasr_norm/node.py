"""Open ASR Leaderboard-compatible text normalization wrappers.

Wraps the vendored normalizer from huggingface/open_asr_leaderboard and
replicates the leaderboard's scoring-side filtering: rows whose normalized
reference is empty or the literal "ignore time segment in scoring" marker are
dropped from both ref and hyp before scoring, matching the official
``is_target_text_in_range`` filter in normalizer/data_utils.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.openasr_norm.normalization_impl import (
    EnglishTextNormalizer,
)

NODE_ID = "normalization/openasr_norm"
NODE_VERSION = "v1"
UPSTREAM_PACKAGE = "open_asr_leaderboard"
UPSTREAM_COMMIT = "8f37837f988e3d641e726b61087b9c761c060013"
UPSTREAM_URL = "https://github.com/huggingface/open_asr_leaderboard/tree/main/normalizer"

IGNORE_MARKER = "ignore time segment in scoring"


def normalize_openasr_asr_files(
    files: KeyTextFiles,
    *,
    language: str = "en",
    profile: str = "english",
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize key-text ASR files with Open ASR Leaderboard text rules."""

    normalizer = _normalizer_for_profile(profile)
    ref_file = _new_temp_file()
    hyp_file = _new_temp_file()
    try:
        ref_rows, dropped_keys = _normalize_key_text_file(files.ref_file, ref_file, normalizer)
        hyp_rows, _ = _normalize_key_text_file(
            files.hyp_file, hyp_file, normalizer, drop_keys=dropped_keys
        )
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
                "dropped_ref_keys": sorted(dropped_keys),
                "drop_policy": "empty_or_ignore_marker_ref_dropped_from_both_sides",
                "normalization": {
                    "backend": "open_asr_leaderboard",
                    "profile": profile,
                    "normalizer_class": _normalizer_class_name(profile),
                    "upstream_package": UPSTREAM_PACKAGE,
                    "upstream_commit": UPSTREAM_COMMIT,
                    "upstream_url": UPSTREAM_URL,
                    "vendored": True,
                },
            },
            internal_stages=("key_text_parse", "openasr_normalize", "empty_ref_filter"),
        ),
    )


def normalize_openasr_text(text: str, *, profile: str = "english") -> str:
    """Normalize one text string with the selected Open ASR Leaderboard profile."""

    return _normalizer_for_profile(profile)(text).strip()


def _normalizer_for_profile(profile: str):
    if profile.lower() == "english":
        return EnglishTextNormalizer()
    raise ValueError(f"Unsupported openasr_norm profile: {profile}")


def _normalizer_class_name(profile: str) -> str:
    if profile.lower() == "english":
        return "EnglishTextNormalizer"
    raise ValueError(f"Unsupported openasr_norm profile: {profile}")


def _normalize_key_text_file(
    input_file: str,
    output_file: str,
    normalizer,
    *,
    drop_keys: set[str] | None = None,
) -> tuple[list[dict[str, str]], set[str]]:
    """Normalize one key-text file, dropping ignored references and their hyps.

    Returns the kept rows and the keys dropped because the normalized
    reference was empty or the leaderboard ignore marker. ``drop_keys``
    replays that set on the hypothesis side so both files stay aligned.
    """
    rows: list[dict[str, str]] = []
    dropped: set[str] = set()
    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if "\t" not in line:
                continue
            key, original_text = line.rstrip("\n").split("\t", 1)
            normalized_text = normalizer(original_text).strip()
            if drop_keys is not None:
                if key in drop_keys:
                    continue
            elif not normalized_text or normalized_text == IGNORE_MARKER:
                # Reference side: replicate the leaderboard scoring filter,
                # which tests the normalized text for emptiness or the
                # ignore marker (is_target_text_in_range on norm_text).
                dropped.add(key)
                continue
            fout.write(f"{key}\t{normalized_text}\n")
            rows.append(
                {
                    "key": key,
                    "original_text": original_text,
                    "normalized_text": normalized_text,
                }
            )
    return rows, dropped


def _new_temp_file() -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    return path
