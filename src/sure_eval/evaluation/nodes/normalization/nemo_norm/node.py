"""Node-local NeMo Arabic text normalization for ASR evaluation."""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
import tempfile
import types
from importlib import metadata
from pathlib import Path
from typing import Any

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.common.node_local_python import (
    build_node_local_env,
    resolve_node_local_python,
)

NODE_ID = "normalization/nemo_norm"
NODE_VERSION = "v1"
PROFILE = "ar_tn"
NEMO_PACKAGE = "nemo_text_processing"
NEMO_VERSION = "1.2.0"
NODE_DIR = Path(__file__).resolve().parent
MODULE_NAME = "sure_eval.evaluation.nodes.normalization.nemo_norm.node"


def normalize_nemo_text(
    text: str,
    *,
    cache_dir: str | None = None,
    overwrite_cache: bool = False,
) -> str:
    """Normalize one Arabic ASR transcript to spoken form with NeMo."""

    payload = _run_node_local_json(
        [
            "--text",
            text,
            "--cache-dir",
            cache_dir or "",
            *( ["--overwrite-cache"] if overwrite_cache else []),
        ]
    )
    return str(payload["normalized_text"])


def normalize_nemo_key_text_files(
    files: KeyTextFiles,
    *,
    cache_dir: str | None = None,
    overwrite_cache: bool = False,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize Arabic reference and hypothesis key-text files to spoken form."""

    payload = _run_node_local_json(
        [
            "--ref-file",
            files.ref_file,
            "--hyp-file",
            files.hyp_file,
            "--cache-dir",
            cache_dir or "",
            *( ["--overwrite-cache"] if overwrite_cache else []),
        ]
    )
    normalized = payload.get("normalized_files")
    trace = payload.get("trace")
    if not isinstance(normalized, dict) or not isinstance(trace, dict):
        raise TypeError(f"{NODE_ID} returned invalid payload: {payload}")
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


def _normalize_text_in_process(
    text: str,
    *,
    cache_dir: str | None,
    overwrite_cache: bool,
) -> str:
    normalizer = _build_normalizer(cache_dir=cache_dir, overwrite_cache=overwrite_cache)
    return normalizer.normalize(text, verbose=False).strip()


def _normalize_file_in_process(input_file: str, output_file: str, *, cache_dir: str | None, overwrite_cache: bool) -> dict[str, int]:
    rows = 0
    empty = 0
    malformed = 0
    normalizer = _build_normalizer(cache_dir=cache_dir, overwrite_cache=overwrite_cache)
    with open(input_file, encoding="utf-8") as source, open(output_file, "w", encoding="utf-8") as target:
        for line in source:
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            if "\t" not in raw:
                malformed += 1
                continue
            key, text = raw.split("\t", 1)
            key = key.strip()
            if not key:
                malformed += 1
                continue
            normalized = normalizer.normalize(text.strip(), verbose=False).strip() if text.strip() else ""
            rows += 1
            empty += int(not normalized)
            target.write(f"{key}\t{normalized}\n")
    if rows == 0 and malformed:
        raise ValueError(f"No key-text rows could be parsed from {input_file!r}")
    return {"num_rows": rows, "num_empty_text_rows": empty, "num_dropped_malformed_rows": malformed}


def _build_normalizer(*, cache_dir: str | None, overwrite_cache: bool):
    _install_cdifflib_compat()
    from nemo_text_processing.text_normalization.normalize import Normalizer

    return Normalizer(
        input_case="cased", lang="ar", cache_dir=cache_dir or None, overwrite_cache=overwrite_cache
    )


def _install_cdifflib_compat() -> None:
    """Satisfy NeMo's eager audio-helper import without its optional C extension."""

    if "cdifflib" in sys.modules:
        return
    module = types.ModuleType("cdifflib")
    module.CSequenceMatcher = difflib.SequenceMatcher
    sys.modules["cdifflib"] = module


def _normalize_files_in_process(ref_file: str, hyp_file: str, *, cache_dir: str | None, overwrite_cache: bool):
    ref_out = _new_temp_file()
    hyp_out = _new_temp_file()
    try:
        ref_stats = _normalize_file_in_process(ref_file, ref_out, cache_dir=cache_dir, overwrite_cache=overwrite_cache)
        hyp_stats = _normalize_file_in_process(hyp_file, hyp_out, cache_dir=cache_dir, overwrite_cache=overwrite_cache)
    except Exception:
        Path(ref_out).unlink(missing_ok=True)
        Path(hyp_out).unlink(missing_ok=True)
        raise
    details = {
        "language": "ar",
        "profile": PROFILE,
        "backend": "NeMo Normalizer",
        "package": NEMO_PACKAGE,
        "package_version": _nemo_version(),
        "pinned_package_version": NEMO_VERSION,
        "input_schema": "key_text_files",
        "output_schema": "key_text_files",
        "ref_file": ref_out,
        "hyp_file": hyp_out,
        "row_stats": {"ref": ref_stats, "hyp": hyp_stats},
    }
    return {
        "normalized_files": {"ref_file": ref_out, "hyp_file": hyp_out},
        "trace": {"details": details, "internal_stages": ["key_text_parse", "ar_tn", "key_text_write"]},
    }


def _run_node_local_json(args: list[str]) -> dict[str, Any]:
    repo_root = NODE_DIR.parents[5]
    try:
        python_runtime = resolve_node_local_python(NODE_DIR, NODE_ID)
    except RuntimeError as exc:
        raise RuntimeError(f"{NODE_ID} requires its node-local environment. Run: sure-eval env setup --node {NODE_ID}") from exc
    env = build_node_local_env(
        repo_src=repo_root / "src",
        extra_pythonpath=python_runtime.extra_pythonpath,
        inherit_pythonpath=python_runtime.inherit_pythonpath,
    )
    completed = subprocess.run(
        [*python_runtime.command_prefix, "-m", MODULE_NAME, *args, "--json"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{NODE_ID} failed with exit code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise TypeError(f"{NODE_ID} returned non-object JSON")
    return payload


def _new_temp_file() -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        path = handle.name
    return path


def _nemo_version() -> str:
    try:
        return metadata.version(NEMO_PACKAGE)
    except metadata.PackageNotFoundError:
        return NEMO_VERSION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")
    parser.add_argument("--ref-file")
    parser.add_argument("--hyp-file")
    parser.add_argument("--cache-dir")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    if args.text is not None:
        payload: dict[str, Any] = {"normalized_text": _normalize_text_in_process(args.text, cache_dir=args.cache_dir or None, overwrite_cache=args.overwrite_cache)}
    elif args.ref_file and args.hyp_file:
        payload = _normalize_files_in_process(args.ref_file, args.hyp_file, cache_dir=args.cache_dir or None, overwrite_cache=args.overwrite_cache)
    else:
        parser.error("provide --text or both --ref-file and --hyp-file")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
