"""SURE adapter for the ETC-Bench model canonicalizer contract."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.common.node_local_python import (
    build_node_local_env,
    resolve_node_local_python,
)

NODE_ID = "normalization/etc_canonicalization"
NODE_VERSION = "v1"
DEFAULT_MODEL = "zhiheng-qian/etc-qwen3.5-0.8b"
ETC_BENCH_VERSION = "2026.10"
NODE_DIR = Path(__file__).resolve().parent
MODULE_NAME = "sure_eval.evaluation.nodes.normalization.etc_canonicalization.runner"
SUPPORTED_PROFILES = ("ar", "bn", "de", "en", "fa", "fr", "hi", "id", "ja", "ru", "vi", "zh")


def _validate_language(language: str) -> str:
    normalized = str(language).lower().strip()
    if normalized not in SUPPORTED_PROFILES:
        supported = ", ".join(SUPPORTED_PROFILES)
        raise ValueError(
            f"Unsupported etc_canonicalization language: {language!r}; supported: {supported}"
        )
    return normalized


def _model_id(model: str | None = None) -> str:
    return str(model or os.environ.get("SURE_ETC_CANONICALIZATION_MODEL") or DEFAULT_MODEL)


def canonicalize_batch(texts: list[str], *, language: str, model: str | None = None) -> list[str]:
    """Canonicalize a batch with one model load in the node-local runtime."""
    normalized_language = _validate_language(language)
    if not texts:
        return []
    try:
        python_runtime = resolve_node_local_python(NODE_DIR, NODE_ID)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{NODE_ID} requires its node-local environment. Run: sure-eval env setup --node {NODE_ID}"
        ) from exc
    repo_root = NODE_DIR.parents[5]
    env = build_node_local_env(
        repo_src=repo_root / "src",
        extra_pythonpath=python_runtime.extra_pythonpath,
        inherit_pythonpath=python_runtime.inherit_pythonpath,
    )
    completed = subprocess.run(
        [
            *python_runtime.command_prefix,
            "-m",
            MODULE_NAME,
            "--language",
            normalized_language,
            "--model",
            _model_id(model),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        input=json.dumps({"texts": texts}, ensure_ascii=False),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{NODE_ID} canonicalizer failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
        predictions = payload["predictions"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"{NODE_ID} returned invalid runner output") from exc
    if not isinstance(predictions, list) or len(predictions) != len(texts) or not all(
        isinstance(prediction, str) for prediction in predictions
    ):
        raise RuntimeError(f"{NODE_ID} returned invalid canonicalization results")
    return [prediction.strip() for prediction in predictions]


def canonicalize_text(text: str, *, language: str, model: str | None = None) -> str:
    return canonicalize_batch([text], language=language, model=model)[0]


def normalize_etc_text(text: str, *, language: str, model: str | None = None) -> str:
    """Canonicalize one text string after validating its ETC language profile."""

    normalized_language = _validate_language(language)
    return canonicalize_text(text, language=normalized_language, model=model)


def normalize_etc_key_text_files(
    files: KeyTextFiles,
    *,
    language: str,
    model: str | None = None,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Canonicalize ref and hyp files independently while preserving keys."""

    normalized_language = _validate_language(language)
    ref_output = None
    try:
        ref_output, ref_rows = _normalize_file(files.ref_file, normalized_language, model=model)
        hyp_output, hyp_rows = _normalize_file(files.hyp_file, normalized_language, model=model)
    except Exception:
        if ref_output:
            Path(ref_output).unlink(missing_ok=True)
        raise
    row_counts = {"ref": len(ref_rows), "hyp": len(hyp_rows)}
    changed_counts = {
        "ref": sum(row["original_text"] != row["normalized_text"] for row in ref_rows),
        "hyp": sum(row["original_text"] != row["normalized_text"] for row in hyp_rows),
    }
    empty_counts = {
        "ref": sum(not row["normalized_text"] for row in ref_rows),
        "hyp": sum(not row["normalized_text"] for row in hyp_rows),
    }
    trace = PipelineNodeResult(
        stage="normalization",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "language": normalized_language,
            "model": _model_id(model),
            "backend": "etc_bench_model_canonicalizer",
            "etc_bench_version": ETC_BENCH_VERSION,
            "input_schema": "key_text_files",
            "output_schema": "key_text_files",
            "row_counts": row_counts,
            "ref_file": ref_output,
            "hyp_file": hyp_output,
            "changed_counts": changed_counts,
            "num_empty_after_normalization": empty_counts,
            "ref_rows": ref_rows,
            "hyp_rows": hyp_rows,
        },
        internal_stages=("key_text_parse", "etc_model_canonicalize", "key_text_write"),
    )
    return KeyTextFiles(ref_file=ref_output, hyp_file=hyp_output), trace


def _normalize_file(
    input_file: str, language: str, *, model: str | None
) -> tuple[str, list[dict[str, str]]]:
    source_rows: list[tuple[str, str]] = []
    with open(input_file, encoding="utf-8") as source:
        for line in source:
            if "\t" in line:
                source_rows.append(tuple(line.rstrip("\n").split("\t", 1)))
    predictions = canonicalize_batch(
        [original for _, original in source_rows], language=language, model=model
    )
    output_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    rows: list[dict[str, str]] = []
    try:
        with output_handle as target:
            for (key, original_text), normalized_text in zip(source_rows, predictions):
                target.write(f"{key}\t{normalized_text}\n")
                rows.append({"key": key, "original_text": original_text, "normalized_text": normalized_text})
    except Exception:
        Path(output_handle.name).unlink(missing_ok=True)
        raise
    return output_handle.name, rows
