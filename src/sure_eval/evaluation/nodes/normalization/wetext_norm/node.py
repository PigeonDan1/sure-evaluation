"""Optional WeTextProcessing normalization wrappers.

This module intentionally lazy-loads WeTextProcessing so the SURE main
environment can import node metadata without installing the node-local project.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult

NODE_ID = "normalization/wetext_norm"
NODE_VERSION = "v1"
PACKAGE_NAME = "WeTextProcessing"
PINNED_PACKAGE_VERSION = "1.2.0"


@dataclass(frozen=True)
class WeTextProfile:
    name: str
    language: str
    direction: str
    module: str
    class_name: str


SUPPORTED_PROFILES: dict[str, WeTextProfile] = {
    "zh_tn": WeTextProfile("zh_tn", "zh", "tn", "tn.chinese.normalizer", "Normalizer"),
    "zh_itn": WeTextProfile("zh_itn", "zh", "itn", "itn.chinese.inverse_normalizer", "InverseNormalizer"),
    "en_tn": WeTextProfile("en_tn", "en", "tn", "tn.english.normalizer", "Normalizer"),
    "en_itn": WeTextProfile("en_itn", "en", "itn", "itn.english.inverse_normalizer", "InverseNormalizer"),
    "ja_tn": WeTextProfile("ja_tn", "ja", "tn", "tn.japanese.normalizer", "Normalizer"),
    "ja_itn": WeTextProfile("ja_itn", "ja", "itn", "itn.japanese.inverse_normalizer", "InverseNormalizer"),
}


def normalize_wetext_text(
    text: str,
    *,
    profile: str,
    cache_dir: str | None = None,
    overwrite_cache: bool = False,
    options: dict[str, Any] | None = None,
) -> str:
    """Normalize one text string with a supported WeTextProcessing profile."""

    normalizer = _normalizer(
        profile,
        cache_dir=cache_dir,
        overwrite_cache=overwrite_cache,
        options=_hashable_options(options or {}),
    )
    return normalizer.normalize(text)


def normalize_wetext_key_text_files(
    files: KeyTextFiles,
    *,
    profile: str,
    cache_dir: str | None = None,
    overwrite_cache: bool = False,
    options: dict[str, Any] | None = None,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize reference and hypothesis key-text files with WeTextProcessing."""

    spec = _profile(profile)
    normalizer = _normalizer(
        spec.name,
        cache_dir=cache_dir,
        overwrite_cache=overwrite_cache,
        options=_hashable_options(options or {}),
    )
    ref_file = _new_temp_file()
    hyp_file = _new_temp_file()
    try:
        ref_rows = _normalize_key_text_file(files.ref_file, ref_file, normalizer)
        hyp_rows = _normalize_key_text_file(files.hyp_file, hyp_file, normalizer)
    except Exception:
        Path(ref_file).unlink(missing_ok=True)
        Path(hyp_file).unlink(missing_ok=True)
        raise

    details = wetext_runtime_details(spec.name)
    details.update(
        {
            "input_schema": "key_text_files",
            "output_schema": "key_text_files",
            "cache_dir": cache_dir,
            "overwrite_cache": overwrite_cache,
            "options": dict(options or {}),
            "ref_file": ref_file,
            "hyp_file": hyp_file,
            "num_rows": {"ref": len(ref_rows), "hyp": len(hyp_rows)},
            "num_empty_after_normalization": {
                "ref": sum(1 for row in ref_rows if not row["normalized_text"]),
                "hyp": sum(1 for row in hyp_rows if not row["normalized_text"]),
            },
            "ref_rows": ref_rows,
            "hyp_rows": hyp_rows,
        }
    )
    return (
        KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details=details,
            internal_stages=("key_text_parse", f"wetext_{spec.direction}", "key_text_write"),
        ),
    )


def wetext_runtime_details(profile: str) -> dict[str, Any]:
    """Return trace-ready runtime metadata without instantiating a normalizer."""

    spec = _profile(profile)
    package_version = _package_version(PACKAGE_NAME)
    return {
        "node_id": NODE_ID,
        "version": NODE_VERSION,
        "profile": spec.name,
        "language": spec.language,
        "direction": spec.direction,
        "package": PACKAGE_NAME,
        "package_version": package_version,
        "pinned_package_version": PINNED_PACKAGE_VERSION,
        "pynini_version": _package_version("pynini"),
        "normalizer_class": f"{spec.module}.{spec.class_name}",
    }


@lru_cache(maxsize=16)
def _normalizer(
    profile: str,
    *,
    cache_dir: str | None,
    overwrite_cache: bool,
    options: tuple[tuple[str, Any], ...],
):
    spec = _profile(profile)
    module = __import__(spec.module, fromlist=[spec.class_name])
    normalizer_cls = getattr(module, spec.class_name)
    kwargs = dict(options)
    kwargs["overwrite_cache"] = overwrite_cache
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return normalizer_cls(**kwargs)


def _profile(profile: str) -> WeTextProfile:
    normalized = profile.lower().strip()
    try:
        return SUPPORTED_PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_PROFILES))
        raise ValueError(f"Unsupported wetext_norm profile {profile!r}; supported: {supported}") from exc


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _hashable_options(options: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(options.items()))


def _normalize_key_text_file(input_file: str, output_file: str, normalizer) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if "\t" not in line:
                continue
            key, original_text = line.rstrip("\n").split("\t", 1)
            normalized_text = normalizer.normalize(original_text)
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
