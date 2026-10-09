"""SURE adapter for the bundled, unmodified XNorm text normalizer."""

from __future__ import annotations

import sys
import json
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.common.node_local_python import (
    build_node_local_env,
    resolve_node_local_python,
)

NODE_ID = "normalization/xnorm"
NODE_VERSION = "v1"
XNORM_VERSION = "1.5.0"
XNORM_LICENSE = "Apache-2.0"
NODE_DIR = Path(__file__).resolve().parent
_RUNTIME_ROOT = NODE_DIR / "xnorm_vendor"
MODULE_NAME = "sure_eval.evaluation.nodes.normalization.xnorm.runner"


_SUPPORTED_PROFILES = (
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "he",
    "hi",
    "hu",
    "hy",
    "id",
    "it",
    "ja",
    "ko",
    "mr",
    "pt",
    "ru",
    "sv",
    "th",
    "vi",
    "zh",
)


def _xnorm_api() -> Callable[..., Any]:
    if str(_RUNTIME_ROOT) not in sys.path:
        sys.path.insert(0, str(_RUNTIME_ROOT))
    try:
        from text_normalization.api import compile_pipeline
    except ImportError as exc:
        raise RuntimeError(
            f"{NODE_ID} requires its bundled XNorm runtime and dependencies. "
            f"Run: sure-eval env setup --node {NODE_ID}"
        ) from exc
    return compile_pipeline


def supported_profiles() -> tuple[str, ...]:
    """Return the language codes registered by XNorm."""

    return _SUPPORTED_PROFILES


SUPPORTED_PROFILES = supported_profiles()


def _validate_language(language: str) -> str:
    normalized = str(language).lower().strip()
    if normalized not in SUPPORTED_PROFILES:
        supported = ", ".join(SUPPORTED_PROFILES)
        raise ValueError(f"Unsupported xnorm language: {language!r}; supported: {supported}")
    return normalized


@lru_cache(maxsize=None)
def _pipeline(language: str) -> Any:
    compile_pipeline = _xnorm_api()
    return compile_pipeline(language, mode="asr_eval")


def normalize_xnorm_text(text: str, *, language: str) -> str:
    """Normalize one string using XNorm's compiled language pipeline."""

    normalized_language = _validate_language(language)
    return str(_pipeline(normalized_language).normalize(text).text_norm).strip()


def normalize_xnorm_key_text_files(
    files: KeyTextFiles,
    *,
    language: str,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize key-text reference and hypothesis files with XNorm."""

    normalized_language = _validate_language(language)
    return _normalize_key_text_files_node_local(files, normalized_language)


def _normalize_key_text_files_in_process(
    files: KeyTextFiles,
    language: str,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Run the adapter in the current interpreter for the node runner."""

    ref_file = _new_temp_file()
    hyp_file = _new_temp_file()
    try:
        ref_rows = _normalize_key_text_file(files.ref_file, ref_file, language)
        hyp_rows = _normalize_key_text_file(files.hyp_file, hyp_file, language)
    except Exception:
        Path(ref_file).unlink(missing_ok=True)
        Path(hyp_file).unlink(missing_ok=True)
        raise

    row_counts = {"ref": len(ref_rows), "hyp": len(hyp_rows)}
    empty_counts = {
        "ref": sum(not row["normalized_text"] for row in ref_rows),
        "hyp": sum(not row["normalized_text"] for row in hyp_rows),
    }
    details = {
        "language": language,
        "profile": language,
        "backend": "xnorm",
        "xnorm_version": XNORM_VERSION,
        "license": XNORM_LICENSE,
        "input_schema": "key_text_files",
        "output_schema": "key_text_files",
        "ref_file": ref_file,
        "hyp_file": hyp_file,
        "row_counts": row_counts,
        "num_rows": row_counts,
        "num_empty_after_normalization": empty_counts,
        "ref_rows": ref_rows,
        "hyp_rows": hyp_rows,
    }
    return (
        KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details=details,
            internal_stages=("key_text_parse", "xnorm_normalize", "key_text_write"),
        ),
    )


def _normalize_key_text_files_node_local(
    files: KeyTextFiles,
    language: str,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    repo_root = NODE_DIR.parents[5]
    try:
        python_runtime = resolve_node_local_python(NODE_DIR, NODE_ID)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{NODE_ID} requires its node-local environment. "
            f"Run: sure-eval env setup --node {NODE_ID}"
        ) from exc
    env = build_node_local_env(
        repo_src=repo_root / "src",
        extra_pythonpath=python_runtime.extra_pythonpath,
        inherit_pythonpath=python_runtime.inherit_pythonpath,
    )
    completed = subprocess.run(
        [*python_runtime.command_prefix, "-m", MODULE_NAME, "--ref-file", files.ref_file,
         "--hyp-file", files.hyp_file, "--language", language],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{NODE_ID} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
        normalized = payload["normalized_files"]
        trace = payload["trace"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"{NODE_ID} returned invalid JSON: {completed.stdout[:500]}") from exc
    return (
        KeyTextFiles(ref_file=str(normalized["ref_file"]), hyp_file=str(normalized["hyp_file"])),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details=dict(trace.get("details") or {}),
            internal_stages=tuple(trace.get("internal_stages") or ()),
        ),
    )


def _normalize_key_text_file(input_file: str, output_file: str, language: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(input_file, encoding="utf-8") as source, open(output_file, "w", encoding="utf-8") as target:
        for line in source:
            if "\t" not in line:
                continue
            key, original_text = line.rstrip("\n").split("\t", 1)
            normalized_text = normalize_xnorm_text(original_text, language=language)
            target.write(f"{key}\t{normalized_text}\n")
            rows.append({"key": key, "original_text": original_text, "normalized_text": normalized_text})
    return rows


def _new_temp_file() -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    return path
