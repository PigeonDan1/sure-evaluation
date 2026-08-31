"""Cosine trial scoring for testset-level speaker verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "scoring/cosine_trial_scores"
NODE_VERSION = "v1"
TRIAL_COLUMNS = ("enroll_key", "test_key", "label", "condition")
LABEL_VALUES = {"target": 1, "nontarget": 0}


@dataclass(frozen=True)
class TrialManifest:
    path: Path
    dataset_id: str
    testset_id: str
    source_dataset_id: str
    trials_path: Path
    trials_sha256: str
    trial_count: int
    target_count: int
    nontarget_count: int


@dataclass(frozen=True)
class SVScoreArtifacts:
    scores_path: Path
    labels_path: Path
    trial_count: int
    target_count: int
    nontarget_count: int
    dataset_id: str
    testset_id: str
    source_dataset_id: str
    embedding_count: int
    extra_embedding_count: int
    embedding_dimension: int

    def scores(self) -> np.memmap:
        return np.memmap(self.scores_path, dtype=np.float32, mode="r", shape=(self.trial_count,))

    def labels(self) -> np.memmap:
        return np.memmap(self.labels_path, dtype=np.uint8, mode="r", shape=(self.trial_count,))


def load_trial_manifest(path: str | Path) -> TrialManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sure.sv.trial_manifest.v1":
        raise ValueError(
            "Unsupported SV trial manifest schema: " f"{payload.get('schema_version')!r}"
        )
    trials_value = payload.get("trials_file")
    if not trials_value:
        raise ValueError("SV trial manifest is missing trials_file")
    trials_path = Path(str(trials_value))
    if not trials_path.is_absolute():
        trials_path = manifest_path.parent / trials_path
    trials_path = trials_path.resolve()
    if not trials_path.is_file():
        raise FileNotFoundError(f"SV trial file does not exist: {trials_path}")
    expected_hash = str(payload.get("trials_sha256") or "")
    if not expected_hash:
        raise ValueError("SV trial manifest is missing trials_sha256")
    actual_hash = _sha256_file(trials_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"SV trial fingerprint mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return TrialManifest(
        path=manifest_path,
        dataset_id=str(payload.get("dataset_id") or "unknown"),
        testset_id=str(payload.get("testset_id") or payload.get("dataset_id") or "unknown"),
        source_dataset_id=str(payload.get("source_dataset_id") or "unknown"),
        trials_path=trials_path,
        trials_sha256=expected_hash,
        trial_count=int(payload.get("trial_count") or 0),
        target_count=int(payload.get("target_count") or 0),
        nontarget_count=int(payload.get("nontarget_count") or 0),
    )


def score_cosine_trials(
    *, sample_output: str | Path, trial_manifest: str | Path, work_dir: str | Path
) -> tuple[SVScoreArtifacts, PipelineNodeResult]:
    manifest = load_trial_manifest(trial_manifest)
    output_path = Path(sample_output).resolve()
    if not output_path.is_file():
        raise FileNotFoundError(f"SV sample output does not exist: {output_path}")
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    key_to_index, dimension = _scan_embedding_file(output_path)
    embedding_count = len(key_to_index)
    embeddings_path = work_path / "embeddings.float32.mmap"
    embeddings = np.memmap(
        embeddings_path,
        dtype=np.float32,
        mode="w+",
        shape=(embedding_count, dimension),
    )
    for key, vector in _iter_embedding_rows(output_path):
        index = key_to_index[key]
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError(f"SV embedding for {key!r} has zero norm")
        embeddings[index] = vector / norm
    embeddings.flush()

    if manifest.trial_count <= 0:
        raise ValueError("SV trial manifest must declare a positive trial_count")
    scores_path = work_path / "trial_scores.float32.mmap"
    labels_path = work_path / "trial_labels.uint8.mmap"
    scores = np.memmap(scores_path, dtype=np.float32, mode="w+", shape=(manifest.trial_count,))
    labels = np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(manifest.trial_count,))
    used_embedding_indexes: set[int] = set()
    missing_keys: set[str] = set()
    target_count = 0
    nontarget_count = 0
    trial_count = 0
    for trial_count, (enroll_key, test_key, label, _) in enumerate(
        _iter_trial_rows(manifest.trials_path), start=1
    ):
        if trial_count > manifest.trial_count:
            raise ValueError(
                f"SV trial file has more rows than manifest trial_count={manifest.trial_count}"
            )
        enroll_index = key_to_index.get(enroll_key)
        test_index = key_to_index.get(test_key)
        if enroll_index is None:
            missing_keys.add(enroll_key)
        if test_index is None:
            missing_keys.add(test_key)
        if missing_keys:
            if len(missing_keys) >= 20:
                break
            continue
        used_embedding_indexes.add(enroll_index)
        used_embedding_indexes.add(test_index)
        row_index = trial_count - 1
        scores[row_index] = np.float32(np.dot(embeddings[enroll_index], embeddings[test_index]))
        labels[row_index] = LABEL_VALUES[label]
        if label == "target":
            target_count += 1
        else:
            nontarget_count += 1
    if missing_keys:
        preview = ", ".join(sorted(missing_keys)[:20])
        raise ValueError(f"SV sample output is missing required embedding key(s): {preview}")
    if trial_count != manifest.trial_count:
        raise ValueError(
            f"SV trial count mismatch: manifest={manifest.trial_count}, actual={trial_count}"
        )
    if target_count != manifest.target_count or nontarget_count != manifest.nontarget_count:
        raise ValueError(
            "SV trial label counts do not match the manifest: "
            f"target={target_count}/{manifest.target_count}, "
            f"nontarget={nontarget_count}/{manifest.nontarget_count}"
        )
    scores.flush()
    labels.flush()
    del embeddings
    embeddings_path.unlink(missing_ok=True)

    artifacts = SVScoreArtifacts(
        scores_path=scores_path,
        labels_path=labels_path,
        trial_count=trial_count,
        target_count=target_count,
        nontarget_count=nontarget_count,
        dataset_id=manifest.dataset_id,
        testset_id=manifest.testset_id,
        source_dataset_id=manifest.source_dataset_id,
        embedding_count=embedding_count,
        extra_embedding_count=embedding_count - len(used_embedding_indexes),
        embedding_dimension=dimension,
    )
    return artifacts, PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "backend": "cosine",
            "result": {
                "dataset_id": manifest.dataset_id,
                "testset_id": manifest.testset_id,
                "source_dataset_id": manifest.source_dataset_id,
                "trial_count": trial_count,
                "target_count": target_count,
                "nontarget_count": nontarget_count,
                "embedding_count": embedding_count,
                "extra_embedding_count": artifacts.extra_embedding_count,
                "embedding_dimension": dimension,
                "trials_sha256": manifest.trials_sha256,
            },
        },
        internal_stages=(
            "embedding_validation",
            "l2_normalization",
            "trial_alignment",
            "cosine_scoring",
        ),
    )


def _scan_embedding_file(path: Path) -> tuple[dict[str, int], int]:
    key_to_index: dict[str, int] = {}
    dimension: int | None = None
    for key, vector in _iter_embedding_rows(path):
        if key in key_to_index:
            raise ValueError(f"Duplicate SV embedding key: {key!r}")
        if dimension is None:
            dimension = int(vector.size)
        elif int(vector.size) != dimension:
            raise ValueError(
                f"Inconsistent SV embedding dimension for {key!r}: "
                f"expected {dimension}, got {vector.size}"
            )
        key_to_index[key] = len(key_to_index)
    if not key_to_index or dimension is None:
        raise ValueError("SV sample output does not contain any embeddings")
    return key_to_index, dimension


def _iter_embedding_rows(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid SV output JSON at {path}:{line_number}") from exc
            key = payload.get("key") or payload.get("sample_id")
            if not isinstance(key, str) or not key:
                raise ValueError(f"Missing SV embedding key at {path}:{line_number}")
            result: Any = payload.get("result", payload)
            embedding = result.get("embedding") if isinstance(result, dict) else None
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"Missing SV embedding vector at {path}:{line_number}")
            try:
                vector = np.asarray(embedding, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid SV embedding at {path}:{line_number}") from exc
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError(f"SV embedding must be a non-empty vector at {path}:{line_number}")
            if not np.isfinite(vector).all():
                raise ValueError(f"SV embedding contains NaN or infinity at {path}:{line_number}")
            declared_dimension = result.get("dimension") if isinstance(result, dict) else None
            if declared_dimension is not None and int(declared_dimension) != int(vector.size):
                raise ValueError(f"SV embedding dimension field mismatch at {path}:{line_number}")
            yield key, vector


def _iter_trial_rows(path: Path) -> Iterator[tuple[str, str, str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        first_data_row = True
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t") if "\t" in stripped else stripped.split()
            if first_data_row and tuple(parts) == TRIAL_COLUMNS:
                first_data_row = False
                continue
            first_data_row = False
            if len(parts) < 3:
                raise ValueError(f"Malformed SV trial row at {path}:{line_number}")
            label = parts[2].lower()
            if label not in LABEL_VALUES:
                raise ValueError(f"Unsupported SV label {parts[2]!r} at {path}:{line_number}")
            condition = parts[3] if len(parts) >= 4 else "default"
            yield parts[0], parts[1], label, condition


def _sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
