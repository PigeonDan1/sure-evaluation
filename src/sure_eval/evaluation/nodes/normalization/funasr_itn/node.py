"""FunASR InverseTextNormalization (ITN) node for ASR evaluation.

The fun_text_processing package is fetched at an immutable Git revision during
env setup and cached under this node directory.
This module lazily loads it in a node-local subprocess so the SURE main
environment can import node metadata without installing pynini.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import sure_eval

from sure_eval.evaluation.cache import get_cache_dir
from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.common.node_local_python import (
    build_node_local_env,
    resolve_node_local_python,
)

NODE_ID = "normalization/funasr_itn"
NODE_VERSION = "v1"
NODE_DIR = Path(__file__).resolve().parent
MODULE_NAME = "sure_eval.evaluation.nodes.normalization.funasr_itn.node"
SOURCE_LOCK_FILE = NODE_DIR / "source_lock.json"
RUNTIME_DIR = NODE_DIR / "runtime"
MARKER_FILE = RUNTIME_DIR / "funasr_revision.json"
_FUNASR_SRC = RUNTIME_DIR


@dataclass(frozen=True)
class FunasrItnProfile:
    name: str
    language: str


SUPPORTED_PROFILES: dict[str, FunasrItnProfile] = {
    "zh": FunasrItnProfile("zh", "zh"),
    "en": FunasrItnProfile("en", "en"),
    "ja": FunasrItnProfile("ja", "ja"),
    "es": FunasrItnProfile("es", "es"),
    "fr": FunasrItnProfile("fr", "fr"),
    "de": FunasrItnProfile("de", "de"),
    "ko": FunasrItnProfile("ko", "ko"),
    "ru": FunasrItnProfile("ru", "ru"),
    "pt": FunasrItnProfile("pt", "pt"),
    "vi": FunasrItnProfile("vi", "vi"),
    "id": FunasrItnProfile("id", "id"),
    "tl": FunasrItnProfile("tl", "tl"),
}


def _load_source_lock() -> dict[str, str]:
    payload = json.loads(SOURCE_LOCK_FILE.read_text(encoding="utf-8"))
    required = ("repository", "revision", "subdirectory", "source_tree", "license_file")
    missing = [key for key in required if not str(payload.get(key) or "").strip()]
    if missing:
        raise RuntimeError(f"Invalid {SOURCE_LOCK_FILE.name}; missing: {', '.join(missing)}")
    return {key: str(payload[key]) for key in required}


def _source_metadata() -> dict[str, str]:
    lock = _load_source_lock()
    try:
        marker = json.loads(MARKER_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{NODE_ID} source is not prepared. Run: sure-eval env setup --node {NODE_ID}"
        ) from exc
    if not isinstance(marker, dict):
        raise RuntimeError(f"{NODE_ID} source marker must contain a JSON object")
    for key in ("repository", "revision", "subdirectory", "source_tree"):
        if marker.get(key) != lock[key]:
            raise RuntimeError(
                f"{NODE_ID} source marker mismatch for {key}: "
                f"expected {lock[key]!r}, got {marker.get(key)!r}. Re-run env setup with --force."
            )
    package_init = RUNTIME_DIR / lock["subdirectory"] / "__init__.py"
    license_file = RUNTIME_DIR / lock["license_file"]
    if not package_init.is_file() or not license_file.is_file():
        raise RuntimeError(
            f"{NODE_ID} runtime source is incomplete. Re-run env setup with --force."
        )
    return {
        "repository": lock["repository"],
        "revision": lock["revision"],
        "subdirectory": lock["subdirectory"],
        "source_tree": lock["source_tree"],
        "license": "Apache-2.0",
    }


def _check_funasr_src() -> None:
    _source_metadata()


def normalize_funasr_text(text: str, *, profile: str) -> str:
    _profile(profile)
    _check_funasr_src()
    return _normalize_funasr_text_node_local(text, profile=profile)


def normalize_funasr_files(
    files: KeyTextFiles,
    *,
    profile: str,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    _profile(profile)
    _check_funasr_src()
    return _normalize_funasr_files_node_local(files, profile=profile)


def _normalize_funasr_text_in_process(text: str, *, profile: str) -> str:
    spec = _profile(profile)
    normalizer = _normalizer(spec.name)
    return _normalize_text(normalizer, text)


def _normalize_funasr_files_in_process(
    files: KeyTextFiles,
    *,
    profile: str,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    spec = _profile(profile)
    normalizer = _normalizer(spec.name)

    ref_file = _new_temp_file()
    hyp_file = _new_temp_file()
    try:
        ref_stats = _normalize_key_text_file(files.ref_file, ref_file, normalizer)
        hyp_stats = _normalize_key_text_file(files.hyp_file, hyp_file, normalizer)
    except Exception:
        Path(ref_file).unlink(missing_ok=True)
        Path(hyp_file).unlink(missing_ok=True)
        raise

    details = funasr_runtime_details(spec.name)
    details.update(
        {
            "input_schema": "key_text_files",
            "output_schema": "key_text_files",
            "ref_file": ref_file,
            "hyp_file": hyp_file,
            "row_stats": {"ref": ref_stats, "hyp": hyp_stats},
            "num_rows": {"ref": ref_stats["num_rows"], "hyp": hyp_stats["num_rows"]},
            "num_empty_after_normalization": {
                "ref": ref_stats["num_empty_after_normalization"],
                "hyp": hyp_stats["num_empty_after_normalization"],
            },
        }
    )
    return (
        KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details=details,
            internal_stages=("key_text_parse", "funasr_itn", "key_text_write"),
        ),
    )


def _normalize_funasr_files_node_local(
    files: KeyTextFiles,
    *,
    profile: str,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    payload = _run_node_local_json(
        [
            "--ref-file",
            files.ref_file,
            "--hyp-file",
            files.hyp_file,
            "--profile",
            profile,
        ]
    )
    normalized = payload.get("normalized_files")
    trace = payload.get("trace")
    if not isinstance(normalized, dict) or not isinstance(trace, dict):
        raise RuntimeError(f"{NODE_ID} returned invalid payload: {payload}")
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


def _normalize_funasr_text_node_local(text: str, *, profile: str) -> str:
    payload = _run_node_local_json(
        [
            "--text",
            text,
            "--profile",
            profile,
        ]
    )
    return str(payload["normalized_text"])


def _run_node_local_json(args: list[str]) -> dict[str, Any]:
    try:
        python_runtime = resolve_node_local_python(NODE_DIR, NODE_ID)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{NODE_ID} requires its node-local environment. "
            f"Run: sure-eval env setup --node {NODE_ID}"
        ) from exc

    extra_pythonpath = (*python_runtime.extra_pythonpath, str(_FUNASR_SRC))
    package_parent = Path(sure_eval.__file__).resolve().parent.parent
    env = build_node_local_env(
        repo_src=package_parent,
        extra_pythonpath=extra_pythonpath,
        inherit_pythonpath=python_runtime.inherit_pythonpath,
    )

    completed = subprocess.run(
        [*python_runtime.command_prefix, "-m", MODULE_NAME, *args, "--json"],
        cwd=NODE_DIR,
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
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{NODE_ID} did not return JSON: {completed.stdout[:500]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{NODE_ID} returned non-object JSON: {completed.stdout[:500]}")
    return payload


def funasr_runtime_details(profile: str) -> dict[str, Any]:
    spec = _profile(profile)
    source = _source_metadata()
    return {
        "node_id": NODE_ID,
        "version": NODE_VERSION,
        "profile": spec.name,
        "language": spec.language,
        "direction": "itn",
        "package": "fun_text_processing (FunASR)",
        "source": source,
        "runtime_versions": {
            "python": platform.python_version(),
            **{
                package: _distribution_version(package)
                for package in ("pynini", "joblib", "tqdm", "regex", "inflect")
            },
        },
        "fst_cache_managed": True,
        "normalizer_class": f"InverseNormalizer(lang={spec.language})",
    }


@lru_cache(maxsize=16)
def _normalizer(profile: str):
    from fun_text_processing.inverse_text_normalization.inverse_normalize import (
        InverseNormalizer,
    )

    source = _source_metadata()
    cache_dir = get_cache_dir(
        "nodes",
        "normalization__funasr_itn",
        source["revision"],
        f"pynini-{_distribution_version('pynini')}",
        profile,
    )
    return InverseNormalizer(lang=profile, cache_dir=str(cache_dir))


def _distribution_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _profile(profile: str) -> FunasrItnProfile:
    normalized = profile.lower().strip()
    try:
        return SUPPORTED_PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_PROFILES))
        raise ValueError(
            f"Unsupported funasr_itn profile {profile!r}; supported: {supported}"
        ) from exc


def _normalize_text(normalizer: Any, text: str) -> str:
    if not text:
        return ""
    with redirect_stdout(io.StringIO()):
        return str(normalizer.inverse_normalize(text, verbose=False))


def _normalize_key_text_file(input_file: str, output_file: str, normalizer: Any) -> dict[str, int]:
    seen_keys: set[str] = set()
    num_rows = 0
    num_blank_lines = 0
    num_empty_text_rows = 0
    num_empty_after_normalization = 0
    with (
        open(input_file, encoding="utf-8") as fin,
        open(output_file, "w", encoding="utf-8") as fout,
    ):
        for line_number, line in enumerate(fin, start=1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                num_blank_lines += 1
                continue
            if "\t" not in raw:
                raise ValueError(
                    f"Malformed key-text row in {input_file!r} at line {line_number}: "
                    "expected <key><TAB><text>"
                )
            key, original_text = raw.split("\t", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Empty key in {input_file!r} at line {line_number}")
            if key in seen_keys:
                raise ValueError(f"Duplicate key {key!r} in {input_file!r} at line {line_number}")
            seen_keys.add(key)
            if not original_text:
                num_empty_text_rows += 1
            normalized_text = _normalize_text(normalizer, original_text)
            if not normalized_text:
                num_empty_after_normalization += 1
            fout.write(f"{key}\t{normalized_text}\n")
            num_rows += 1
    if num_rows == 0:
        raise ValueError(f"No <key><TAB><text> rows found in {input_file!r}")
    return {
        "num_rows": num_rows,
        "num_blank_lines": num_blank_lines,
        "num_empty_text_rows": num_empty_text_rows,
        "num_empty_after_normalization": num_empty_after_normalization,
        "num_dropped_malformed_rows": 0,
    }


def _new_temp_file() -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize text with FunASR ITN.")
    parser.add_argument("--text")
    parser.add_argument("--ref-file")
    parser.add_argument("--hyp-file")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    if bool(args.text is not None) == bool(args.ref_file or args.hyp_file):
        parser.error("provide either --text or both --ref-file/--hyp-file")
    if bool(args.ref_file) != bool(args.hyp_file):
        parser.error("--ref-file and --hyp-file must be provided together")

    if args.text is not None:
        if args.json_output:
            with redirect_stdout(sys.stderr):
                normalized_text = _normalize_funasr_text_in_process(
                    text=args.text, profile=args.profile
                )
        else:
            normalized_text = _normalize_funasr_text_in_process(
                text=args.text, profile=args.profile
            )
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "profile": args.profile,
            "normalized_text": normalized_text,
            "details": funasr_runtime_details(args.profile),
        }
    else:
        files = KeyTextFiles(ref_file=str(args.ref_file), hyp_file=str(args.hyp_file))
        if args.json_output:
            with redirect_stdout(sys.stderr):
                normalized_files, trace = _normalize_funasr_files_in_process(
                    files, profile=args.profile
                )
        else:
            normalized_files, trace = _normalize_funasr_files_in_process(
                files, profile=args.profile
            )
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "profile": args.profile,
            "normalized_files": {
                "ref_file": normalized_files.ref_file,
                "hyp_file": normalized_files.hyp_file,
            },
            "trace": {
                "stage": trace.stage,
                "node_id": trace.node_id,
                "version": trace.version,
                "details": trace.details,
                "internal_stages": list(trace.internal_stages),
            },
        }

    if args.json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        if args.text is not None:
            print(payload["normalized_text"])
        else:
            print(json.dumps(payload["normalized_files"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
