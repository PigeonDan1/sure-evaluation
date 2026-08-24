"""FunASR InverseTextNormalization (ITN) node for ASR evaluation.

The fun_text_processing package is fetched from GitHub during env setup
(via build_funasr_itn.sh) and cached under this node directory.
This module lazily loads it in a node-local subprocess so the SURE main
environment can import node metadata without installing pynini.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.common.node_local_python import (
    build_node_local_env,
    resolve_node_local_python,
)

NODE_ID = "normalization/funasr_itn"
NODE_VERSION = "v1"
NODE_DIR = Path(__file__).resolve().parent
MODULE_NAME = "sure_eval.evaluation.nodes.normalization.funasr_itn.node"

# fun_text_processing is vendored under this node directory
_FUNASR_SRC = NODE_DIR


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


def _check_funasr_src() -> None:
    pkg_dir = _FUNASR_SRC / "fun_text_processing"
    if not pkg_dir.is_dir():
        raise RuntimeError(
            f"fun_text_processing source not found at {pkg_dir}. "
            f"Run: sure-eval env setup --node {NODE_ID}"
        )


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
    return normalizer.inverse_normalize(text, verbose=False)


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
        ref_rows = _normalize_key_text_file(files.ref_file, ref_file, normalizer)
        hyp_rows = _normalize_key_text_file(files.hyp_file, hyp_file, normalizer)
    except Exception:
        Path(ref_file).unlink(missing_ok=True)
        Path(hyp_file).unlink(missing_ok=True)
        raise

    details = {
        "node_id": NODE_ID,
        "version": NODE_VERSION,
        "profile": spec.name,
        "language": spec.language,
        "input_schema": "key_text_files",
        "output_schema": "key_text_files",
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
    repo_root = NODE_DIR.parents[5]
    try:
        python_runtime = resolve_node_local_python(NODE_DIR, NODE_ID)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{NODE_ID} requires its node-local environment. "
            f"Run: sure-eval env setup --node {NODE_ID}"
        ) from exc

    extra_pythonpath = (*python_runtime.extra_pythonpath, str(_FUNASR_SRC))
    env = build_node_local_env(
        repo_src=repo_root / "src",
        extra_pythonpath=extra_pythonpath,
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
    return {
        "node_id": NODE_ID,
        "version": NODE_VERSION,
        "profile": spec.name,
        "language": spec.language,
        "direction": "itn",
        "package": "fun_text_processing (FunASR)",
        "source_path": str(_FUNASR_SRC),
        "normalizer_class": f"InverseNormalizer(lang={spec.language})",
    }


@lru_cache(maxsize=16)
def _normalizer(profile: str):
    from fun_text_processing.inverse_text_normalization.inverse_normalize import (
        InverseNormalizer,
    )

    return InverseNormalizer(lang=profile, cache_dir=None)


def _profile(profile: str) -> FunasrItnProfile:
    normalized = profile.lower().strip()
    try:
        return SUPPORTED_PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_PROFILES))
        raise ValueError(f"Unsupported funasr_itn profile {profile!r}; supported: {supported}") from exc


def _normalize_key_text_file(input_file: str, output_file: str, normalizer) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if "\t" not in line:
                continue
            key, original_text = line.rstrip("\n").split("\t", 1)
            normalized_text = normalizer.inverse_normalize(original_text, verbose=False)
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
                normalized_text = _normalize_funasr_text_in_process(text=args.text, profile=args.profile)
        else:
            normalized_text = _normalize_funasr_text_in_process(text=args.text, profile=args.profile)
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
                normalized_files, trace = _normalize_funasr_files_in_process(files, profile=args.profile)
        else:
            normalized_files, trace = _normalize_funasr_files_in_process(files, profile=args.profile)
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
