"""Node-local subprocess providers for heavy audio scoring models."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sure_eval.evaluation.nodes.common.node_local_python import (
    build_node_local_env,
    resolve_node_local_python,
)


def _repo_root_from_node_dir(node_dir: Path) -> Path:
    return node_dir.parents[5]


def _run_node_json(
    *,
    node_id: str,
    node_dir: Path,
    module_name: str,
    args: list[str],
) -> dict[str, Any]:
    repo_root = _repo_root_from_node_dir(node_dir)
    python_runtime = resolve_node_local_python(node_dir, node_id)
    env = build_node_local_env(
        repo_src=repo_root / "src",
        extra_pythonpath=python_runtime.extra_pythonpath,
        inherit_pythonpath=python_runtime.inherit_pythonpath,
    )
    completed = subprocess.run(
        [*python_runtime.command_prefix, "-m", module_name, *args, "--json"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{node_id} scoring failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{node_id} did not return JSON: {completed.stdout[:500]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{node_id} returned non-object JSON: {completed.stdout[:500]}")
    return payload


@dataclass(frozen=True)
class NodeLocalSpeakerProvider:
    """Speaker similarity provider backed by a scoring node's local uv env."""

    node_id: str
    node_dir: Path
    device: str = "cuda"

    def __call__(self, prediction: str, reference: str, **kwargs: Any) -> dict[str, Any]:
        payload = _run_node_json(
            node_id=self.node_id,
            node_dir=self.node_dir,
            module_name=f"sure_eval.evaluation.nodes.scoring.{self.node_id.split('/', 1)[1]}.node",
            args=[
                "--prediction-audio",
                prediction,
                "--reference-audio",
                reference,
                "--device",
                self.device,
            ],
        )
        result = payload.get("result", payload)
        if not isinstance(result, dict):
            raise RuntimeError(f"{self.node_id} returned invalid result payload: {payload}")
        return dict(result)

    def score_batch(
        self,
        rows: list[tuple[str, str, str]],
        *,
        metric_name: str,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        chunk_size = _speaker_batch_size(self.node_id)
        if chunk_size and len(rows) > chunk_size:
            per_sample: list[dict[str, Any]] = []
            for chunk in _chunk_rows(rows, chunk_size):
                per_sample.extend(self._score_batch_resilient(chunk))
            return per_sample
        return self._score_batch_resilient(rows)

    def _score_batch_once(self, rows: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
        return self._score_batch_once_on_device(rows, self.device)

    def _score_batch_once_on_device(self, rows: list[tuple[str, str, str]], device: str) -> list[dict[str, Any]]:
        input_path = _write_speaker_rows(rows)
        try:
            payload = _run_node_json(
                node_id=self.node_id,
                node_dir=self.node_dir,
                module_name=f"sure_eval.evaluation.nodes.scoring.{self.node_id.split('/', 1)[1]}.node",
                args=[
                    "--input-jsonl",
                    str(input_path),
                    "--device",
                    device,
                ],
            )
        finally:
            input_path.unlink(missing_ok=True)
        per_sample = _extract_per_sample_rows(self.node_id, payload, expected=len(rows))
        return per_sample

    def _score_batch_resilient(self, rows: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
        try:
            return self._score_batch_once(rows)
        except RuntimeError as exc:
            if not _should_retry_eres2net_cpu(self.node_id, self.device, exc):
                raise
            if len(rows) > 1:
                midpoint = len(rows) // 2
                return [
                    *self._score_batch_resilient(rows[:midpoint]),
                    *self._score_batch_resilient(rows[midpoint:]),
                ]
            return self._score_batch_once_on_device(rows, "cpu")


@dataclass(frozen=True)
class NodeLocalMOSProvider:
    """MOS provider backed by a scoring node's local uv env."""

    node_id: str
    node_dir: Path
    device: str = "cuda"

    def __call__(self, prediction: str, reference: str = "", **kwargs: Any) -> dict[str, Any]:
        payload = _run_node_json(
            node_id=self.node_id,
            node_dir=self.node_dir,
            module_name=f"sure_eval.evaluation.nodes.scoring.{self.node_id.split('/', 1)[1]}.node",
            args=[
                "--prediction-audio",
                prediction,
                "--device",
                self.device,
            ],
        )
        result = payload.get("result", payload)
        if not isinstance(result, dict):
            raise RuntimeError(f"{self.node_id} returned invalid result payload: {payload}")
        return dict(result)

    def score_batch(
        self,
        rows: list[tuple[str, str]],
        *,
        metric_name: str,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        chunk_size = _mos_batch_size(self.node_id)
        if chunk_size and len(rows) > chunk_size:
            per_sample: list[dict[str, Any]] = []
            for chunk in _chunk_mos_rows(rows, chunk_size):
                per_sample.extend(self._score_batch_once(chunk))
            return per_sample
        return self._score_batch_once(rows)

    def _score_batch_once(self, rows: list[tuple[str, str]]) -> list[dict[str, Any]]:
        input_path = _write_mos_rows(rows)
        try:
            payload = _run_node_json(
                node_id=self.node_id,
                node_dir=self.node_dir,
                module_name=f"sure_eval.evaluation.nodes.scoring.{self.node_id.split('/', 1)[1]}.node",
                args=[
                    "--input-jsonl",
                    str(input_path),
                    "--device",
                    self.device,
                ],
            )
        finally:
            input_path.unlink(missing_ok=True)
        per_sample = _extract_per_sample_rows(self.node_id, payload, expected=len(rows))
        return per_sample


def _write_speaker_rows(rows: list[tuple[str, str, str]]) -> Path:
    input_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    try:
        input_path = Path(input_file.name)
        for key, prediction, reference in rows:
            input_file.write(
                json.dumps(
                    {
                        "key": key,
                        "prediction_audio": prediction,
                        "reference_audio": reference,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        input_file.close()
    return input_path


def _speaker_batch_size(node_id: str) -> int:
    env_name = "SURE_EVAL_ERES2NET_BATCH_SIZE" if node_id == "scoring/eres2net_sim" else "SURE_EVAL_NODE_LOCAL_SPEAKER_BATCH_SIZE"
    default_value = "0"
    raw_value = os.environ.get(env_name, default_value).strip()
    if not raw_value:
        return 0
    try:
        batch_size = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    if batch_size < 0:
        raise RuntimeError(f"{env_name} must be non-negative, got {batch_size}")
    return batch_size


def _mos_batch_size(node_id: str) -> int:
    env_name = "SURE_EVAL_NODE_LOCAL_MOS_BATCH_SIZE"
    default_value = "128" if node_id == "scoring/dnsmos" else "0"
    raw_value = os.environ.get(env_name, default_value).strip()
    if not raw_value:
        return 0
    try:
        batch_size = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    if batch_size < 0:
        raise RuntimeError(f"{env_name} must be non-negative, got {batch_size}")
    return batch_size


def _chunk_rows(
    rows: list[tuple[str, str, str]],
    chunk_size: int,
) -> list[list[tuple[str, str, str]]]:
    return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]


def _chunk_mos_rows(
    rows: list[tuple[str, str]],
    chunk_size: int,
) -> list[list[tuple[str, str]]]:
    return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]


def _should_retry_eres2net_cpu(node_id: str, device: str, exc: RuntimeError) -> bool:
    if node_id != "scoring/eres2net_sim" or not device.startswith("cuda"):
        return False
    if not _env_flag("SURE_EVAL_ERES2NET_ALLOW_CPU_FALLBACK"):
        return False
    message = str(exc).lower()
    return "outofmemoryerror" in message or "out of memory" in message or "cuda oom" in message


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _write_mos_rows(rows: list[tuple[str, str]]) -> Path:
    input_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    try:
        input_path = Path(input_file.name)
        for key, prediction in rows:
            input_file.write(
                json.dumps(
                    {
                        "key": key,
                        "prediction_audio": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        input_file.close()
    return input_path


def _extract_per_sample_rows(node_id: str, payload: dict[str, Any], *, expected: int) -> list[dict[str, Any]]:
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        raise RuntimeError(f"{node_id} returned invalid result payload: {payload}")
    per_sample = result.get("per_sample")
    if not isinstance(per_sample, list):
        raise RuntimeError(f"{node_id} batch payload is missing result.per_sample: {payload}")
    if len(per_sample) != expected:
        raise RuntimeError(f"{node_id} returned {len(per_sample)} row(s) for {expected} input row(s)")
    normalized: list[dict[str, Any]] = []
    for row in per_sample:
        if not isinstance(row, dict):
            raise RuntimeError(f"{node_id} returned a non-object per-sample row: {row!r}")
        normalized.append(dict(row))
    return normalized
