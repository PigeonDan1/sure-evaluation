"""NIST SCTK sclite scoring wrapper.

This node treats SCTK as an external binary dependency. The main Python
environment only resolves and calls ``sclite`` when this optional scorer is
selected explicitly.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sure_eval.evaluation.cache import get_cache_dir
from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult

NODE_ID = "scoring/sctk_sclite"
NODE_VERSION = "v1"
SCTK_REPO = "https://github.com/usnistgov/SCTK"
PINNED_SCTK_COMMIT = "9688a26882a688132a5e414cadcb4c19b6fffaba"
DEFAULT_CACHE_ROOT = get_cache_dir("tools", "sctk")
ENV_SCLITE_BIN = "SURE_EVAL_SCLITE_BIN"
ENV_SCTK_ROOT = "SURE_EVAL_SCTK_ROOT"
INTERNAL_STAGES = ("key_text_parse", "trn_materialize", "sclite", "sclite_report_parse")

_SCORES_RE = re.compile(
    r"Scores:\s*\(#C\s+#S\s+#D\s+#I\)\s*"
    r"(?P<cor>\d+)\s+(?P<sub>\d+)\s+(?P<del>\d+)\s+(?P<ins>\d+)",
    re.IGNORECASE,
)
_PERCENT_TOTAL_ERROR_RE = re.compile(
    r"Percent\s+Total\s+Error\s*=\s*(?P<percent>[0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScliteBinary:
    path: str
    source: str
    searched_paths: tuple[str, ...]


def score_sctk_sclite_wer(
    files: KeyTextFiles,
    *,
    sclite_bin: str | None = None,
    keep_artifacts: bool = False,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score normalized key-text files with SCTK sclite WER."""

    return _score_sclite(files, metric="wer", sclite_bin=sclite_bin, keep_artifacts=keep_artifacts)


def score_sctk_sclite_cer(
    files: KeyTextFiles,
    *,
    sclite_bin: str | None = None,
    keep_artifacts: bool = False,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score normalized key-text files with SCTK sclite CER using NOASCII tokenization."""

    return _score_sclite(files, metric="cer", sclite_bin=sclite_bin, keep_artifacts=keep_artifacts)


def resolve_sclite_binary(sclite_bin: str | None = None) -> ScliteBinary:
    """Resolve the sclite binary without mutating the main Python environment."""

    searched: list[str] = []
    if sclite_bin:
        candidate = Path(sclite_bin).expanduser()
        searched.append(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return ScliteBinary(path=str(candidate), source="argument", searched_paths=tuple(searched))

    env_bin = os.environ.get(ENV_SCLITE_BIN)
    if env_bin:
        candidate = Path(env_bin).expanduser()
        searched.append(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return ScliteBinary(path=str(candidate), source=ENV_SCLITE_BIN, searched_paths=tuple(searched))

    path_candidate = shutil.which("sclite")
    searched.append("PATH:sclite")
    if path_candidate:
        return ScliteBinary(path=path_candidate, source="PATH", searched_paths=tuple(searched))

    roots: list[Path] = []
    env_root = os.environ.get(ENV_SCTK_ROOT)
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend(
        [
            DEFAULT_CACHE_ROOT,
            Path(__file__).resolve().parent / ".local" / "sctk",
        ]
    )
    for root in roots:
        candidate = root / PINNED_SCTK_COMMIT / "bin" / "sclite"
        searched.append(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return ScliteBinary(path=str(candidate), source=str(root), searched_paths=tuple(searched))

    raise RuntimeError(
        "NIST SCTK sclite binary was not found. "
        f"Set {ENV_SCLITE_BIN} to an executable sclite path, add sclite to PATH, "
        f"or build SCTK with src/sure_eval/evaluation/nodes/scoring/sctk_sclite/build_sctk.sh. "
        f"Searched: {', '.join(searched)}. Current PATH={os.environ.get('PATH', '')!r}"
    )


def _score_sclite(
    files: KeyTextFiles,
    *,
    metric: str,
    sclite_bin: str | None,
    keep_artifacts: bool,
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    binary = resolve_sclite_binary(sclite_bin)
    ref_rows = _read_key_text_file(files.ref_file)
    hyp_rows = _read_key_text_file(files.hyp_file)
    _validate_key_sets(ref_rows, hyp_rows)

    work_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if keep_artifacts:
        work_dir = Path(tempfile.mkdtemp(prefix="sure-sclite-"))
    else:
        work_dir_obj = tempfile.TemporaryDirectory(prefix="sure-sclite-")
        work_dir = Path(work_dir_obj.name)

    try:
        id_map = _materialize_trn_files(ref_rows, hyp_rows, work_dir)
        report_root = "score"
        command = [
            binary.path,
            "-r",
            str(work_dir / "ref.trn"),
            "trn",
            "-h",
            str(work_dir / "hyp.trn"),
            "trn",
            "-i",
            "rm",
            "-o",
            "dtl",
            "prf",
            "sum",
            "-O",
            str(work_dir),
            "-n",
            report_root,
        ]
        if metric == "cer":
            command.extend(["-c", "NOASCII"])

        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        report_texts = _collect_report_texts(work_dir, completed)
        if completed.returncode != 0:
            raise RuntimeError(
                "sclite failed with return code "
                f"{completed.returncode}. Command: {command!r}. "
                f"stdout={completed.stdout[-2000:]!r}; stderr={completed.stderr[-2000:]!r}"
            )
        result = _parse_sclite_result(report_texts, metric=metric)

        details: dict[str, Any] = {
            "backend": "NIST SCTK sclite",
            "metric": metric,
            "input_schema": "normalized_key_text_files",
            "intermediate_format": "trn",
            "binary": {
                "path": binary.path,
                "source": binary.source,
                "searched_paths": list(binary.searched_paths),
                "sctk_repo": SCTK_REPO,
                "pinned_sctk_commit": PINNED_SCTK_COMMIT,
            },
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
            "num_rows": len(ref_rows),
            "id_map": id_map,
            "result": result,
        }
        if keep_artifacts:
            details["artifact_dir"] = str(work_dir)
            details["side_outputs"] = _side_outputs(work_dir)
        else:
            details["report_excerpt"] = _report_excerpt(report_texts)

        return (
            files,
            PipelineNodeResult(
                stage="scoring",
                node_id=NODE_ID,
                version=NODE_VERSION,
                details=details,
                internal_stages=INTERNAL_STAGES,
            ),
        )
    finally:
        if work_dir_obj is not None:
            work_dir_obj.cleanup()


def _read_key_text_file(path: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            if "\t" not in stripped:
                raise ValueError(f"Expected key-tab-text format in {path}:{line_number}")
            key, text = stripped.split("\t", 1)
            if not key:
                raise ValueError(f"Empty key in {path}:{line_number}")
            if key in rows:
                raise ValueError(f"Duplicate key {key!r} in {path}:{line_number}")
            rows[key] = text
    if not rows:
        raise ValueError(f"No key-text rows found in {path}")
    return rows


def _validate_key_sets(ref_rows: dict[str, str], hyp_rows: dict[str, str]) -> None:
    ref_keys = set(ref_rows)
    hyp_keys = set(hyp_rows)
    if ref_keys == hyp_keys:
        return
    missing_in_hyp = sorted(ref_keys - hyp_keys)
    extra_in_hyp = sorted(hyp_keys - ref_keys)
    raise ValueError(
        "sctk_sclite requires identical ref/hyp key sets; "
        f"missing_in_hyp={missing_in_hyp[:10]}, extra_in_hyp={extra_in_hyp[:10]}"
    )


def _materialize_trn_files(ref_rows: dict[str, str], hyp_rows: dict[str, str], work_dir: Path) -> dict[str, str]:
    id_map = {key: f"sure-{index:06d}" for index, key in enumerate(ref_rows, start=1)}
    _write_trn(work_dir / "ref.trn", ref_rows, id_map)
    _write_trn(work_dir / "hyp.trn", hyp_rows, id_map)
    return dict(id_map)


def _write_trn(path: Path, rows: dict[str, str], id_map: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, text in rows.items():
            normalized_text = _one_line_text(text)
            handle.write(f"{normalized_text} ({id_map[key]})\n")


def _one_line_text(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def _collect_report_texts(work_dir: Path, completed: subprocess.CompletedProcess[str]) -> dict[str, str]:
    reports = {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    for suffix in ("dtl", "sys", "prf", "raw", "sum"):
        paths = sorted(work_dir.glob(f"*.{suffix}"))
        if not paths:
            continue
        reports[suffix] = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths)
    return reports


def _parse_sclite_result(report_texts: dict[str, str], *, metric: str) -> dict[str, Any]:
    counts = _parse_dtl_scores(report_texts)
    if counts is None:
        percent = _parse_percent_total_error(report_texts)
        if percent is None:
            available = ", ".join(sorted(key for key, value in report_texts.items() if value))
            raise ValueError(f"Could not parse sclite result from reports: {available}")
        score = percent / 100.0
        return {
            "metric_name": metric,
            "score": score,
            metric: score,
            f"{metric}_percent": percent,
            "aggregation": "sclite_percent_total_error",
        }

    ref_units = counts["cor"] + counts["sub"] + counts["del"]
    errors = counts["sub"] + counts["del"] + counts["ins"]
    score = 0.0 if ref_units == 0 else errors / ref_units
    percent = score * 100
    return {
        "metric_name": metric,
        "score": score,
        metric: score,
        f"{metric}_percent": percent,
        "all": ref_units,
        "cor": counts["cor"],
        "sub": counts["sub"],
        "del": counts["del"],
        "ins": counts["ins"],
        "errors": errors,
        "num_sentences": counts["num_sentences"],
        "sentence_errors": counts["sentence_errors"],
        "aggregation": "sctk_sclite_dtl_scores",
    }


def _parse_dtl_scores(report_texts: dict[str, str]) -> dict[str, int] | None:
    total = {"cor": 0, "sub": 0, "del": 0, "ins": 0, "num_sentences": 0, "sentence_errors": 0}
    for text in report_texts.values():
        for match in _SCORES_RE.finditer(text):
            row = {name: int(match.group(name)) for name in ("cor", "sub", "del", "ins")}
            for name, value in row.items():
                total[name] += value
            total["num_sentences"] += 1
            if row["sub"] or row["del"] or row["ins"]:
                total["sentence_errors"] += 1
    if total["num_sentences"] == 0:
        return None
    return total


def _parse_percent_total_error(report_texts: dict[str, str]) -> float | None:
    for text in report_texts.values():
        match = _PERCENT_TOTAL_ERROR_RE.search(text)
        if match:
            return float(match.group("percent"))
    return None


def _side_outputs(work_dir: Path) -> dict[str, str]:
    outputs = {"ref_trn": str(work_dir / "ref.trn"), "hyp_trn": str(work_dir / "hyp.trn")}
    for path in sorted(work_dir.iterdir()):
        if path.suffix and path.name not in {"ref.trn", "hyp.trn"}:
            outputs[path.name] = str(path)
    return outputs


def _report_excerpt(report_texts: dict[str, str]) -> str:
    for key in ("dtl", "sys", "prf", "stdout", "stderr"):
        text = report_texts.get(key, "")
        if text:
            return text[:4000]
    return ""
