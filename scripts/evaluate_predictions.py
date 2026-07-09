#!/usr/bin/env python3
"""
Evaluate prepared prediction files against canonical SURE-EVAL datasets.

This script is deterministic by design:
- dataset resolution goes through DatasetManager
- metric selection goes through SOTA baseline first
- optional result recording goes through RPSManager
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import string
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sure_eval.core.logging import configure_logging, get_logger
from sure_eval.evaluation.cache import get_cache_dir
from sure_eval.evaluation.scripts import run_task
from sure_eval.reports.sota_manager import SOTAManager

configure_logging(level="INFO")
logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_REPO_PREFIX = "/workspace/sure-eval/"


class _SimpleDatasetManager:
    """Dataset resolver for minimal metric images without pydantic config deps."""

    def __init__(self, jsonl_dir: Path | None = None) -> None:
        self.jsonl_dir = jsonl_dir or (REPO_ROOT / "data" / "datasets" / "sure_benchmark" / "jsonl")

    def normalize_dataset_name(self, name: str) -> str:
        if (self.jsonl_dir / f"{name}.jsonl").exists():
            return name
        normalized = name.lower().replace("-", "_")
        if (self.jsonl_dir / f"{normalized}.jsonl").exists():
            return normalized
        return name

    def expand_dataset_names(self, dataset_names: list[str] | tuple[str, ...]) -> list[str]:
        return [self.normalize_dataset_name(name) for name in dataset_names]

    def get_jsonl_path(self, dataset_name: str) -> Path:
        return self.jsonl_dir / f"{self.normalize_dataset_name(dataset_name)}.jsonl"

    def download_and_convert(self, dataset_name: str) -> Path:
        path = self.get_jsonl_path(dataset_name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset JSONL not found in minimal evaluation image: {path}")
        return path


@dataclass(frozen=True)
class TTSSample:
    """Local TTS sample record for minimal metric images."""

    prediction_audio: str
    reference_text: str
    reference_audio: str = ""
    language: str = "en"
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _NullRPSManager:
    database = None


def _build_dataset_and_rps_managers(config_path: str | None):
    """Load full managers when available; fall back for metric-only images."""
    if os.environ.get("SURE_EVAL_MINIMAL_DATASET_MANAGER", "").lower() in {"1", "true", "yes"}:
        return _SimpleDatasetManager(), _NullRPSManager()
    try:
        from sure_eval.core.config import Config
        from sure_eval.datasets import DatasetManager
        from sure_eval.evaluation.rps import RPSManager

        cfg = Config.from_yaml(config_path) if config_path else Config.from_env()
        return DatasetManager(cfg), RPSManager(cfg)
    except ModuleNotFoundError as exc:
        if exc.name != "pydantic":
            raise
        logger.warning(
            "pydantic unavailable; using minimal dataset resolver and skipping RPS database writes",
            error=str(exc),
        )
        return _SimpleDatasetManager(), _NullRPSManager()

LOWER_IS_BETTER_METRICS = {
    "wer",
    "cer",
    "mer",
    "der",
    "cpwer",
    "tts_wer",
    "tts_cer",
    "vc_wer",
    "vc_cer",
}
PERCENT_DISPLAY_METRICS = LOWER_IS_BETTER_METRICS | {"accuracy"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_prediction_map(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            key, value = line.split("\t", 1)
        else:
            parts = line.split(None, 1)
            key = parts[0]
            value = parts[1] if len(parts) > 1 else ""
        predictions[key] = value
    return predictions


def _metric_slug(metric: str) -> str:
    return metric.lower().replace("/", "_").replace(" ", "_")


def _display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _localize_path(path: str | Path | None) -> Path:
    value = "" if path is None else str(path)
    candidate = Path(value)
    if candidate.exists() or not value.startswith(WORKSPACE_REPO_PREFIX):
        return candidate
    return REPO_ROOT / value[len(WORKSPACE_REPO_PREFIX) :]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    if not output:
        return None
    return output.splitlines()[0]


def _git_state() -> dict[str, Any]:
    commit = _command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    dirty = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if completed.returncode == 0:
            dirty = bool(completed.stdout.strip())
    except Exception:
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _runtime_versions(pipeline_description: dict[str, Any] | None = None) -> dict[str, Any]:
    packages = {
        name: _package_version(name)
        for name in ("sure-eval", "torch", "transformers", "funasr", "sacrebleu", "meeteval", "PyYAML")
    }
    nodes = []
    if pipeline_description:
        for node in pipeline_description.get("nodes") or []:
            nodes.append(
                {
                    "node_id": node.get("node_id"),
                    "stage": node.get("stage"),
                    "manifest_version": node.get("version"),
                    "manifest_path": node.get("manifest_path"),
                }
            )
    return {
        "sure_eval": {
            "package_version": packages.pop("sure-eval", None),
            "git": _git_state(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": packages,
        "tools": {
            "ffmpeg": _command_output(["ffmpeg", "-version"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "pipeline_nodes": nodes,
    }


def _default_metric_for_task(task: str, language: str) -> str:
    return (
        "accuracy" if task in {"SER", "GR", "SLU"}
        else "bleu" if task == "S2TT"
        else "der" if task == "SD"
        else "cpwer" if task == "SA-ASR"
        else "tts_cer" if task == "TTS" and language == "zh"
        else "tts_wer" if task == "TTS"
        else "mer" if task == "ASR" and language == "cs"
        else "wer" if task == "ASR" and language == "en"
        else "cer"
    )


def _is_chinese_family_language(language: str) -> bool:
    return str(language).lower().startswith(("zh", "cmn", "yue"))


def _dataset_task_language(dataset_manager: DatasetManager, dataset_name: str) -> tuple[str, str]:
    jsonl_path = dataset_manager.get_jsonl_path(dataset_name)
    if not jsonl_path.exists():
        jsonl_path = dataset_manager.download_and_convert(dataset_name)
    samples = load_jsonl(jsonl_path)
    if not samples:
        raise ValueError(f"Dataset has no samples: {dataset_name}")
    first = samples[0]
    return str(first.get("task", "ASR")).upper(), str(first.get("language", "auto")).lower()


def _metric_applies_to_task_language(metric: str | None, task: str, language: str) -> bool:
    if metric is None:
        return True
    metric_name = metric.lower()
    if task == "TTS":
        if metric_name == "tts_cer":
            return _is_chinese_family_language(language)
        if metric_name == "tts_wer":
            return not _is_chinese_family_language(language)
    if task == "VC":
        if metric_name == "vc_cer":
            return _is_chinese_family_language(language)
        if metric_name == "vc_wer":
            return not _is_chinese_family_language(language)
    return True


def build_tts_runtime(
    *,
    metrics: tuple[str, ...],
    language: str,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build TTS runtime through the canonical evaluation audio-runtime layer."""

    if os.environ.get("SURE_TTS_AUDIO_RUNTIME", "").lower() == "in_process":
        return _build_in_process_tts_runtime(metrics=metrics, language=language, device=device, cache_dir=cache_dir)

    from sure_eval.evaluation.audio_runtime import build_tts_runtime as build_audio_tts_runtime

    return build_audio_tts_runtime(
        metrics=metrics,
        language=language,
        device=device,
        cache_dir=cache_dir,
    )


def _prepend_node_site_packages(node_id: str) -> None:
    node_path = REPO_ROOT / "src" / "sure_eval" / "evaluation" / "nodes" / Path(node_id)
    site_packages = (
        node_path
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


def _build_in_process_tts_runtime(
    *,
    metrics: tuple[str, ...],
    language: str,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    requested = {metric.lower() for metric in metrics}
    cache_root = Path(cache_dir) if cache_dir is not None else get_cache_dir("tts-metrics")
    transcribers: dict[str, Any] = {}
    speaker_providers: dict[str, Any] = {}
    mos_providers: dict[str, Any] = {}

    if requested & {"tts_wer", "tts_cer"}:
        if language.lower().startswith(("zh", "cmn", "yue")):
            _prepend_node_site_packages("transcription/paraformer_zh")
            from sure_eval.evaluation.nodes.transcription.paraformer_zh.node import DEFAULT_CACHE_DIR
            from sure_eval.evaluation.nodes.transcription.common.providers import ParaformerZHTranscriber

            transcribers["zh"] = ParaformerZHTranscriber(device=device, cache_dir=DEFAULT_CACHE_DIR)
        else:
            _prepend_node_site_packages("transcription/whisper_large_v3")
            from sure_eval.evaluation.nodes.transcription.whisper_large_v3.node import DEFAULT_CACHE_DIR
            from sure_eval.evaluation.nodes.transcription.common.providers import WhisperLargeV3Transcriber

            transcribers["en"] = WhisperLargeV3Transcriber(device=device, cache_dir=DEFAULT_CACHE_DIR)

    speaker_metrics = {metric.removeprefix("sim/") for metric in requested if metric.startswith("sim/")}
    if speaker_metrics:
        from sure_eval.evaluation.nodes.scoring.common.speaker_providers import EmbeddingSpeakerSimilarityProvider

        speaker_cache = cache_root / "speaker"
        if "wavlm-large" in speaker_metrics:
            _prepend_node_site_packages("scoring/wavlm_large_sim")
            from sure_eval.evaluation.nodes.scoring.common.speaker_providers import WavLMSpeakerEmbeddingProvider

            speaker_providers["wavlm-large"] = EmbeddingSpeakerSimilarityProvider(
                WavLMSpeakerEmbeddingProvider(device=device, cache_dir=speaker_cache),
                backend="wavlm-large-cosine",
            )
        if "ecapa-tdnn" in speaker_metrics:
            _prepend_node_site_packages("scoring/ecapa_tdnn_sim")
            from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ECAPATDNNEmbeddingProvider

            speaker_providers["ecapa-tdnn"] = EmbeddingSpeakerSimilarityProvider(
                ECAPATDNNEmbeddingProvider(device=device, cache_dir=speaker_cache),
                backend="speechbrain-ecapa-tdnn-cosine",
            )
        if "eres2net" in speaker_metrics:
            _prepend_node_site_packages("scoring/eres2net_sim")
            from sure_eval.evaluation.nodes.scoring.common.speaker_providers import (
                ERes2NetEmbeddingProvider,
                ERes2NetSimilarityProvider,
            )

            speaker_providers["eres2net"] = ERes2NetSimilarityProvider(
                device=device,
                cache_dir=speaker_cache,
                embedding_provider=ERes2NetEmbeddingProvider(device=device, cache_dir=speaker_cache),
            )

    mos_metrics = requested & {"dnsmos", "wv-mos", "utmos"}
    if mos_metrics:
        mos_cache = cache_root / "mos"
        if "dnsmos" in mos_metrics:
            _prepend_node_site_packages("scoring/dnsmos")
            from sure_eval.evaluation.nodes.scoring.common.mos_providers import DNSMOSProvider

            mos_providers["dnsmos"] = DNSMOSProvider(cache_dir=mos_cache)
        if "wv-mos" in mos_metrics:
            _prepend_node_site_packages("scoring/wv_mos")
            from sure_eval.evaluation.nodes.scoring.common.mos_providers import WVMOSProvider

            mos_providers["wv-mos"] = WVMOSProvider(cache_dir=mos_cache, device=device)
        if "utmos" in mos_metrics:
            _prepend_node_site_packages("scoring/utmos")
            from sure_eval.evaluation.nodes.scoring.common.mos_providers import UTMOSProvider

            mos_providers["utmos"] = UTMOSProvider(cache_dir=mos_cache, device=device)

    return {
        "transcribers": transcribers,
        "speaker_providers": speaker_providers,
        "mos_providers": mos_providers,
    }


def _metric_display(metric: str, score: float) -> dict[str, Any]:
    unit = "fraction" if metric.lower() in PERCENT_DISPLAY_METRICS else "score"
    if unit == "fraction":
        return {"unit": unit, "display": f"{score * 100:.6f}%"}
    return {"unit": unit, "display": f"{score:.6f}"}


def _mean_numeric(rows: Any, key: str) -> float | None:
    values: list[float] = []
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _first_row_value(rows: Any, key: str) -> Any | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get(key) is not None:
            return row[key]
    return None


def _score_key_for_metric(metric: str) -> str:
    normalized = metric.lower()
    if normalized.endswith("_wer") or normalized == "wer":
        return "wer"
    if normalized.endswith("_cer") or normalized == "cer":
        return "cer"
    if normalized.startswith("sim/"):
        return "similarity"
    if normalized == "dnsmos":
        return "OVRL"
    if normalized in {"wv-mos", "utmos"}:
        return "mos"
    return normalized


def _metric_summary_details(result: dict[str, Any]) -> dict[str, Any]:
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    metric = str(result.get("metric"))
    if isinstance(details.get("per_sample"), list) or isinstance(details.get("rows"), list):
        return details

    report_path_value = result.get("metric_report_path")
    if report_path_value:
        report = _read_json(Path(str(report_path_value)))
        report_details = report.get("details") if isinstance(report.get("details"), dict) else {}
        report_results = report_details.get("results") if isinstance(report_details.get("results"), dict) else {}
        artifact_details = report_results.get(metric)
        if isinstance(artifact_details, dict):
            merged = dict(artifact_details)
            for key, value in details.items():
                if key not in {"per_sample", "rows"}:
                    merged.setdefault(key, value)
            return merged

    return details


def _compact_metric_result(result: dict[str, Any]) -> dict[str, Any]:
    """Build the public metric summary for evaluation_payload.json.

    The full metric reports may contain sample-level rows. Those stay under
    metrics/ and sample_reports/; the final payload only carries dataset-level
    summary values.
    """

    metric = str(result["metric"])
    score = float(result["score"])
    num_samples = result.get("num_samples")
    details = _metric_summary_details(result)
    score_key = _score_key_for_metric(metric)
    compact: dict[str, Any] = {
        "metric_name": metric,
        "score": score,
        "num_samples": num_samples,
        "score_key": score_key,
    }

    normalized = metric.lower()
    if normalized.startswith("sim/"):
        rows = details.get("per_sample") or details.get("rows")
        compact.update(
            {
                "similarity": score,
                "backend": _first_row_value(rows, "backend"),
                "source": details.get("source"),
            }
        )
    elif normalized == "dnsmos":
        rows = details.get("per_sample") or details.get("rows")
        compact["OVRL"] = score
        compact["mos"] = score
        for source_key, public_key in (
            ("SIG", "mean_SIG"),
            ("BAK", "mean_BAK"),
            ("P808_MOS", "mean_P808_MOS"),
            ("OVRL_raw", "mean_OVRL_raw"),
            ("SIG_raw", "mean_SIG_raw"),
            ("BAK_raw", "mean_BAK_raw"),
        ):
            value = _mean_numeric(rows, source_key)
            if value is not None:
                compact[public_key] = value
        compact["source"] = details.get("source")
    elif normalized in {"wv-mos", "utmos"}:
        compact.update({"mos": score, "source": details.get("source")})
    else:
        compact[score_key] = score
        for key in ("aggregation", "asr_metric", "asr_pipeline_id", "mean_sample_score"):
            if key in details:
                compact[key] = details[key]

    return {key: value for key, value in compact.items() if value is not None}


def _higher_is_better(metric: str, baseline: Any | None) -> bool:
    if baseline is not None and getattr(baseline, "metric", None) == metric:
        return bool(getattr(baseline, "higher_is_better", False))
    return metric.lower() not in LOWER_IS_BETTER_METRICS


def _baseline_key(sota_manager: SOTAManager, dataset: str) -> str:
    get_baseline = getattr(sota_manager, "get_baseline", None)
    if not callable(get_baseline):
        return dataset
    if get_baseline(dataset):
        return dataset
    projection_base = dataset.split("__", 1)[0]
    if projection_base != dataset and get_baseline(projection_base):
        return projection_base
    return dataset


def _baseline_for_dataset(sota_manager: SOTAManager, dataset: str) -> Any | None:
    get_baseline = getattr(sota_manager, "get_baseline", None)
    if not callable(get_baseline):
        return None
    return get_baseline(_baseline_key(sota_manager, dataset))


def _baseline_payload(sota_manager: SOTAManager, dataset: str) -> dict[str, Any] | None:
    baseline_key = _baseline_key(sota_manager, dataset)
    baseline = _baseline_for_dataset(sota_manager, baseline_key)
    if not baseline:
        return None
    return {
        "source": _display_path(sota_manager.sota_file),
        "dataset": baseline.dataset,
        "evaluated_dataset": dataset,
        "match": "exact" if baseline_key == dataset else "projection_base_alias",
        "metric": baseline.metric,
        "score": baseline.score,
        "score_unit": "percent" if baseline.metric.lower() in LOWER_IS_BETTER_METRICS | {"accuracy"} else "score",
        "higher_is_better": baseline.higher_is_better,
        "sota_model": baseline.sota_model,
        "description": baseline.description,
    }


def _rps_payload(sota_manager: SOTAManager, dataset: str, score: float, rps: Any) -> dict[str, Any]:
    if isinstance(rps, dict):
        return rps
    baseline = _baseline_for_dataset(sota_manager, dataset)
    if not baseline:
        return {"value": rps, "status": "missing_baseline"}
    if isinstance(rps, float) and not math.isfinite(rps):
        return {
            "value": None,
            "status": "unbounded_perfect_score",
            "raw_value": "inf",
            "formula": "normalized_baseline / normalized_score",
            "normalized_score": 0.0,
            "normalized_baseline": sota_manager.normalize_baseline_score_for_rps(
                baseline.metric,
                baseline.score,
                higher_is_better=baseline.higher_is_better,
            ),
        }
    normalized_score = sota_manager.normalize_evaluator_score_for_rps(
        baseline.metric,
        score,
        higher_is_better=baseline.higher_is_better,
    )
    normalized_baseline = sota_manager.normalize_baseline_score_for_rps(
        baseline.metric,
        baseline.score,
        higher_is_better=baseline.higher_is_better,
    )
    formula = "normalized_score / normalized_baseline" if baseline.higher_is_better else "normalized_baseline / normalized_score"
    return {
        "value": rps,
        "formula": formula,
        "normalized_score": normalized_score,
        "normalized_baseline": normalized_baseline,
    }


def _write_eval_file(rows: list[str]) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    handle.write("\n".join(rows) + "\n")
    handle.close()
    return handle.name


def _write_annotation_file(rows: list[str], *, suffix: str) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    handle.write("\n".join(row for row in rows if row.strip()) + "\n")
    handle.close()
    return handle.name


def _get_s2tt_source_text(sample: dict[str, Any]) -> str:
    for field in ("source", "src", "source_text", "transcript", "speech_text"):
        value = sample.get(field)
        if value is not None:
            return str(value)
    return ""


def _materialize_meeteval_annotations(
    *,
    samples: list[dict[str, Any]],
    predictions: dict[str, str],
    task: str,
) -> tuple[str, str]:
    suffix = ".rttm" if task == "SD" else ".stm"
    ref_rows = [_annotation_from_sample(sample) for sample in samples]
    hyp_rows = [_annotation_from_prediction(predictions.get(str(sample.get("key", "")), "")) for sample in samples]
    return (
        _write_annotation_file(_flatten_annotation_rows(ref_rows), suffix=suffix),
        _write_annotation_file(_flatten_annotation_rows(hyp_rows), suffix=suffix),
    )


def _annotation_from_sample(sample: dict[str, Any]) -> str | list[str]:
    for field in (
        "annotation",
        "annotation_text",
        "reference_annotation",
        "target_annotation",
        "rttm",
        "stm",
        "target",
    ):
        if sample.get(field) is not None:
            return _annotation_value_to_rows(sample[field])
    segments = sample.get("segments") or sample.get("target_segments") or sample.get("reference_segments")
    if segments is not None:
        return _annotation_value_to_rows(segments)
    return ""


def _annotation_from_prediction(value: str) -> str | list[str]:
    stripped = value.strip()
    if not stripped:
        return ""
    path = Path(stripped)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8").splitlines()
    return _annotation_value_to_rows(stripped)


def _annotation_value_to_rows(value: Any) -> str | list[str]:
    if isinstance(value, list):
        return [_segment_to_annotation_row(item) if isinstance(item, dict) else str(item) for item in value]
    if isinstance(value, dict):
        return _segment_to_annotation_row(value)
    text = str(value)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    return _annotation_value_to_rows(decoded)


def _segment_to_annotation_row(segment: dict[str, Any]) -> str:
    for field in ("line", "annotation", "rttm", "stm"):
        if segment.get(field) is not None:
            return str(segment[field])
    session = segment.get("session_id", segment.get("session", segment.get("recording_id", segment.get("key", "session"))))
    channel = segment.get("channel", "1")
    speaker = segment.get("speaker", segment.get("speaker_id", "speaker"))
    start = float(segment.get("start", segment.get("start_time", 0.0)))
    end_value = segment.get("end", segment.get("end_time"))
    duration_value = segment.get("duration")
    if end_value is not None:
        end = float(end_value)
    elif duration_value is not None:
        end = start + float(duration_value)
    else:
        end = start
    text = segment.get("text", segment.get("transcript", segment.get("words", "")))
    return f"{session} {channel} {speaker} {start:.2f} {end:.2f} {text}"


def _flatten_annotation_rows(rows: list[str | list[str]]) -> list[str]:
    flattened: list[str] = []
    for row in rows:
        if isinstance(row, list):
            candidates = row
        else:
            candidates = str(row).splitlines()
        flattened.extend(candidate.strip() for candidate in candidates if candidate.strip())
    return flattened


def _resolve_prediction_audio(value: str, prediction_path: Path) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    audio_path = Path(stripped)
    if audio_path.is_absolute() and not audio_path.exists():
        workspace_root = Path("/workspace/sure-eval")
        try:
            relative_to_workspace = audio_path.relative_to(workspace_root)
        except ValueError:
            pass
        else:
            remapped = REPO_ROOT / relative_to_workspace
            if remapped.exists():
                audio_path = remapped
    if not audio_path.is_absolute():
        audio_path = prediction_path.parent / audio_path
    return str(audio_path.resolve())


def _tts_samples_from_predictions(
    *,
    samples: list[dict[str, Any]],
    predictions: dict[str, str],
    prediction_path: Path,
) -> list[TTSSample]:
    tts_samples: list[TTSSample] = []
    for sample in samples:
        key = str(sample.get("key", ""))
        prediction_audio = _resolve_prediction_audio(predictions.get(key, ""), prediction_path)
        if not prediction_audio:
            raise ValueError(f"TTS prediction for {key} must be a generated-audio path")
        if not Path(prediction_audio).exists():
            raise FileNotFoundError(f"TTS prediction audio not found for {key}: {prediction_audio}")
        tts_samples.append(
            TTSSample(
                prediction_audio=prediction_audio,
                reference_text=str(sample.get("reference_text") or sample.get("target") or ""),
                reference_audio=str(sample.get("reference_audio") or sample.get("path") or ""),
                language=str(sample.get("language") or "en"),
                sample_id=str(sample.get("sample_id") or key),
                metadata={
                    field: value
                    for field, value in sample.items()
                    if field
                    not in {
                        "sample_id",
                        "reference_text",
                        "reference_audio",
                        "target",
                        "path",
                        "language",
                    }
                },
            )
        )
    return tts_samples


def _describe_evaluation_context(task: str, language: str, metric: str, metric_source: str) -> dict[str, Any]:
    """Describe the dataset-driven post-processing used by the evaluator."""
    context: dict[str, Any] = {
        "task": task,
        "language": language,
        "language_source": "dataset_jsonl",
        "metric": metric,
        "metric_source": metric_source,
    }

    if task == "ASR":
        context.update(
            {
                "postprocessing": "sure_eval.evaluation.scripts.run_task",
                "task_route": "src/sure_eval/evaluation/tasks/asr/routes.yaml",
                "normalization_node": "normalization/aispeech_norm",
                "scoring_node": "scoring/wenet_wer",
                "punctuation_policy": "evaluation-pipeline clean_marks.strip_all_punct compatible",
                "tokenization": "code_switch_mer_wer_cer" if language == "cs" else "character" if metric == "cer" or language == "zh" else "word",
                "case_sensitive": False,
            }
        )
    elif task == "S2TT":
        context.update(
            {
                "postprocessing": "sure_eval.evaluation.scripts.run_task",
                "task_route": "src/sure_eval/evaluation/tasks/s2tt/routes.yaml",
                "scoring_backend": f"scoring/{metric}",
                "scoring_config": "tokenizer_by_language",
            }
        )
    elif task in {"SER", "GR", "SLU"}:
        context.update(
            {
                "postprocessing": "sure_eval.evaluation.scripts.run_task",
                "task_route": "src/sure_eval/evaluation/tasks/slu/routes.yaml" if task == "SLU" else "src/sure_eval/evaluation/tasks/classification/routes.yaml",
                "normalization_node": "normalization/prompt_norm" if task == "SLU" else None,
                "scoring_node": "scoring/classify",
            }
        )
    elif task == "SA-ASR":
        context.update(
            {
                "postprocessing": "sure_eval.evaluation.scripts.run_task",
                "task_route": "src/sure_eval/evaluation/tasks/sa_asr/routes.yaml",
                "scoring_node": "scoring/meeteval",
                "scoring_backend": "meeteval",
                "input_loader": "meeteval.io.load",
                "input_format_policy": "annotation_file_not_key_text",
                "default_collar": 0.5,
                "companion_metrics": ["der"],
            }
        )
    elif task == "SD":
        context.update(
            {
                "postprocessing": "sure_eval.evaluation.scripts.run_task",
                "task_route": "src/sure_eval/evaluation/tasks/sd/routes.yaml",
                "scoring_node": "scoring/meeteval",
                "scoring_backend": "meeteval",
                "input_loader": "meeteval.io.load",
                "input_format_policy": "annotation_file_not_key_text",
                "default_collar": 0.25,
            }
        )

    return context


def evaluate_prediction_file(
    dataset_manager: DatasetManager,
    sota_manager: SOTAManager,
    dataset_name: str,
    prediction_path: Path,
    metric_override: str | None = None,
    metrics_root: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    canonical_name = dataset_manager.normalize_dataset_name(dataset_name)
    jsonl_path = dataset_manager.get_jsonl_path(canonical_name)
    if not jsonl_path.exists():
        jsonl_path = dataset_manager.download_and_convert(canonical_name)

    samples = load_jsonl(jsonl_path)
    predictions = load_prediction_map(prediction_path)
    if not samples:
        raise ValueError(f"Dataset has no samples: {canonical_name}")

    task = samples[0].get("task", "ASR")
    language = samples[0].get("language", "auto")
    metric_source = "cli_override" if metric_override else "sota_baseline"
    baseline_key = _baseline_key(sota_manager, canonical_name)
    metric = metric_override or sota_manager.get_metric(baseline_key)
    if not metric:
        metric = _default_metric_for_task(task, language)
        metric_source = "task_default"

    ref_file: str | None = None
    hyp_file: str | None = None
    if task == "TTS":
        tts_samples = _tts_samples_from_predictions(
            samples=samples,
            predictions=predictions,
            prediction_path=prediction_path,
        )
    elif task in {"SD", "SA-ASR"}:
        ref_file, hyp_file = _materialize_meeteval_annotations(
            samples=samples,
            predictions=predictions,
            task=task,
        )
    else:
        ref_file = _write_eval_file([f"{sample.get('key', '')}\t{sample.get('target', '')}" for sample in samples])
        hyp_file = _write_eval_file([f"{sample.get('key', '')}\t{predictions.get(sample.get('key', ''), '')}" for sample in samples])
    src_file: str | None = None
    if task == "S2TT" and metric == "xcomet_xl":
        src_file = _write_eval_file(
            [f"{sample.get('key', '')}\t{_get_s2tt_source_text(sample)}" for sample in samples]
        )

    report = None
    output_dir: str | None = None
    cleanup_output_dir = False
    try:
        if metrics_root is not None:
            output_path = metrics_root / canonical_name / _metric_slug(metric)
            output_path.mkdir(parents=True, exist_ok=True)
            output_dir = str(output_path)
        else:
            output_dir = str(Path(tempfile.mkdtemp(prefix=f"sure-eval-{task.lower()}-")))
            cleanup_output_dir = True
        if task == "TTS":
            logger.info(
                "Running TTS metric",
                dataset=canonical_name,
                metric=metric,
                output_dir=output_dir,
            )
            tts_runtime = build_tts_runtime(metrics=(metric,), language=language, device=device, cache_dir=None)
            report = run_task(
                "tts",
                samples=tts_samples,
                metrics=(metric,),
                output_dir=output_dir,
                transcribers=tts_runtime["transcribers"],
                speaker_providers=tts_runtime["speaker_providers"],
                mos_providers=tts_runtime["mos_providers"],
            )
            result = report.details["results"][metric]
            logger.info(
                "Completed TTS metric",
                dataset=canonical_name,
                metric=metric,
                output_dir=output_dir,
            )
        elif task == "ASR":
            assert ref_file is not None and hyp_file is not None
            report = run_task(
                "asr",
                ref_file=ref_file,
                hyp_file=hyp_file,
                language=language,
                metric=metric,
                output_dir=output_dir,
            )
            result = report.details["scoring_result"]
        elif task == "S2TT":
            assert ref_file is not None and hyp_file is not None
            report = run_task(
                "s2tt",
                ref_file=ref_file,
                hyp_file=hyp_file,
                language=language,
                metric=metric,
                output_dir=output_dir,
                src_file=src_file,
            )
            result = report.details["scoring_result"]
        elif task in {"SER", "GR"}:
            assert ref_file is not None and hyp_file is not None
            report = run_task(
                task.lower(),
                ref_file=ref_file,
                hyp_file=hyp_file,
                output_dir=output_dir,
            )
            result = report.details["scoring_result"]
        elif task == "SLU":
            assert ref_file is not None and hyp_file is not None
            report = run_task(
                "slu",
                ref_file=ref_file,
                hyp_file=hyp_file,
                prompt_jsonl=str(jsonl_path),
                output_dir=output_dir,
            )
            result = report.details["scoring_result"]
        elif task == "SD":
            assert ref_file is not None and hyp_file is not None
            report = run_task(
                "sd",
                ref_file=ref_file,
                hyp_file=hyp_file,
                output_dir=output_dir,
            )
            result = report.details["scoring_result"]
        elif task == "SA-ASR":
            assert ref_file is not None and hyp_file is not None
            report = run_task(
                "sa-asr",
                ref_file=ref_file,
                hyp_file=hyp_file,
                language=language,
                output_dir=output_dir,
            )
            result = report.details["scoring_result"]
        else:
            from sure_eval.evaluation.sure_evaluator import SUREEvaluator

            assert ref_file is not None and hyp_file is not None
            evaluator = SUREEvaluator(language=language)
            result = evaluator.evaluate(task, ref_file, hyp_file)
    finally:
        if ref_file is not None:
            Path(ref_file).unlink(missing_ok=True)
        if hyp_file is not None:
            Path(hyp_file).unlink(missing_ok=True)
        if src_file is not None:
            Path(src_file).unlink(missing_ok=True)
        if cleanup_output_dir and output_dir is not None:
            shutil.rmtree(output_dir, ignore_errors=True)

    if isinstance(result, dict):
        details = result
    else:
        details = {"score": result}

    if task == "ASR":
        score = details.get(metric, details.get("score", 0.0))
    elif task == "S2TT" and metric == "bleu_char":
        score = details.get("bleu_char", details.get("bleu", details.get("score", 0.0)))
    elif task == "S2TT":
        score = details.get(metric, details.get("score", 0.0))
    elif task in {"SER", "GR", "SLU"}:
        score = details.get("accuracy", details.get("score", 0.0))
    elif task == "SD":
        score = details.get("der", details.get("score", 0.0))
    elif task == "SA-ASR":
        score = details.get("cpwer", details.get("score", 0.0))
    else:
        score = details.get("score", 0.0)

    metric_output_dir = Path(output_dir) if output_dir and not cleanup_output_dir else None
    pipeline_description = _read_json(metric_output_dir / "pipeline_description.json") if metric_output_dir else {}
    metric_report_path = metric_output_dir / "report.json" if metric_output_dir else None
    pipeline_description_path = metric_output_dir / "pipeline_description.json" if metric_output_dir else None
    baseline = _baseline_for_dataset(sota_manager, baseline_key)
    if baseline is not None and baseline.metric != metric:
        rps = {
            "status": "metric_not_comparable_to_baseline",
            "dataset": canonical_name,
            "baseline_dataset": baseline_key,
            "metric": metric,
            "baseline_metric": baseline.metric,
        }
    else:
        rps = sota_manager.calculate_rps(baseline_key, score)

    return {
        "dataset": canonical_name,
        "jsonl_path": str(jsonl_path),
        "prediction_path": str(prediction_path),
        "task": task,
        "language": language,
        "metric": metric,
        "baseline_dataset": baseline_key,
        "score": score,
        "rps": rps,
        "rps_is_unbounded": isinstance(rps, float) and not math.isfinite(rps),
        "num_samples": len(samples),
        "evaluation_context": _describe_evaluation_context(task, language, metric, metric_source),
        "pipeline_id": pipeline_description.get("pipeline_id") or getattr(report, "pipeline_id", None),
        "metric_artifact_dir": str(metric_output_dir) if metric_output_dir else None,
        "metric_report_path": str(metric_report_path) if metric_report_path else None,
        "pipeline_description_path": str(pipeline_description_path) if pipeline_description_path else None,
        "pipeline_description": pipeline_description,
        "details": details,
    }


def _to_strict_jsonable(value: Any) -> Any:
    """Convert Python objects into strict-JSON-safe values."""
    if isinstance(value, dict):
        return {key: _to_strict_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_to_strict_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_protocol_yaml(
    path: Path,
    *,
    run_dir: Path | None,
    tool_name: str,
    protocol_id: str,
    model_dir: Path | None,
    results: list[dict[str, Any]],
) -> None:
    import yaml

    datasets_by_name: dict[str, dict[str, Any]] = {}
    for result in results:
        dataset = datasets_by_name.setdefault(
            result["dataset"],
            {
                "name": result["dataset"],
                "task": result["task"],
                "language": result["language"],
                "metrics": [],
                "jsonl_path": _display_path(result["jsonl_path"]),
            },
        )
        metric = str(result["metric"])
        if metric not in dataset["metrics"]:
            dataset["metrics"].append(metric)
    datasets = []
    for dataset in datasets_by_name.values():
        dataset["metrics"] = sorted(dataset["metrics"])
        datasets.append(dataset)

    payload = {
        "schema": "sure.eval.protocol.v1",
        "protocol_id": protocol_id,
        "run": {
            "run_id": run_dir.name if run_dir else None,
            "run_dir": _display_path(run_dir) if run_dir else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "model": {
            "model_dir": _display_path(model_dir) if model_dir else None,
            "tool_name": tool_name,
            "server_config": _load_model_server_config(model_dir),
        },
        "datasets": datasets,
        "prediction_contract": {
            "contract_path": "docs/agents/main_flow_agent/contracts/prediction_output_contract.md",
            "compatibility_tsv": "predictions/<dataset>.txt",
            "structured_jsonl": "predictions/<dataset>.jsonl",
        },
        "evaluation_policy": {
            "preserve_metric_artifacts": True,
            "metric_artifact_layout": "metrics/<dataset>/<metric_slug>/",
            "report_jsonl_granularity": "one row per dataset metric",
            "require_pipeline_id_match": True,
            "require_prediction_validation": True,
            "rps_baseline_source": "reports/sota/sota_baseline.yaml",
        },
        "artifact_layout": {
            "report_jsonl": "report.jsonl",
            "protocol": "protocol.yaml",
            "metrics": "metrics/<dataset>/<metric_slug>/",
            "sample_reports": "sample_reports/<dataset>/<metric_slug>.jsonl",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _load_model_server_config(model_dir: Path | None) -> dict[str, Any]:
    if not model_dir:
        return {}
    config_yaml = model_dir / "config.yaml"
    if not config_yaml.exists():
        return {}
    try:
        import yaml

        cfg = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    server = dict(cfg.get("server") or {})
    if "env" in server:
        server["env_keys"] = sorted(str(key) for key in (server.get("env") or {}).keys())
        server.pop("env", None)
    return server


def _validation_summary(validation_payload: dict[str, Any], dataset: str) -> dict[str, Any]:
    for result in validation_payload.get("results") or []:
        if result.get("dataset") == dataset:
            return {
                "expected_samples": result.get("expected_samples"),
                "provided_predictions": result.get("provided_predictions"),
                "missing": len(result.get("missing_keys") or []),
                "extra": len(result.get("extra_keys") or []),
                "duplicate": len(result.get("duplicate_keys") or []),
                "empty": len(result.get("empty_prediction_keys") or []),
                "is_valid": result.get("is_valid"),
                "prediction_jsonl_path": result.get("prediction_jsonl_path"),
                "format_used": result.get("format_used"),
            }
    return {}


def _report_jsonl_line(
    *,
    result: dict[str, Any],
    tool_name: str,
    protocol_id: str,
    run_dir: Path | None,
    model_dir: Path | None,
    sota_manager: SOTAManager,
    validation_payload: dict[str, Any],
) -> dict[str, Any]:
    metric = str(result["metric"])
    score = float(result["score"])
    baseline = _baseline_payload(sota_manager, result["dataset"])
    pipeline_description = result.get("pipeline_description") or {}
    nodes = pipeline_description.get("nodes") or []
    metric_display = _metric_display(metric, score)
    return {
        "schema": "sure.eval.report.dataset_metric.v1",
        "run": {
            "run_id": run_dir.name if run_dir else None,
            "protocol_id": protocol_id,
        },
        "model": {
            "model_name": model_dir.name if model_dir else None,
            "model_dir": _display_path(model_dir) if model_dir else None,
            "tool_name": tool_name,
        },
        "dataset": {
            "name": result["dataset"],
            "task": result["task"],
            "language": result["language"],
            "jsonl_path": _display_path(result["jsonl_path"]),
            "num_samples": result["num_samples"],
        },
        "prediction": {
            "file": _display_path(result["prediction_path"]),
            "validation": _validation_summary(validation_payload, result["dataset"]),
        },
        "metric": {
            "name": metric,
            "score": score,
            "unit": metric_display["unit"],
            "display": metric_display["display"],
            "higher_is_better": _higher_is_better(metric, _baseline_for_dataset(sota_manager, result["dataset"])),
        },
        "baseline": baseline,
        "rps": _rps_payload(sota_manager, result["dataset"], score, result.get("rps")),
        "pipeline": {
            "pipeline_id": result.get("pipeline_id"),
            "report_path": _display_path(result.get("metric_report_path")),
            "description_path": _display_path(result.get("pipeline_description_path")),
            "nodes": nodes,
            "conversion_steps": pipeline_description.get("conversion_steps") or [],
        },
        "versions": _runtime_versions(pipeline_description),
        "artifacts": {
            "metric_artifact_dir": _display_path(result.get("metric_artifact_dir")),
            "sample_report": _display_path(result.get("sample_report_path")),
        },
        "status": "success",
    }


def _write_report_jsonl(
    path: Path,
    *,
    results: list[dict[str, Any]],
    tool_name: str,
    protocol_id: str,
    run_dir: Path | None,
    model_dir: Path | None,
    sota_manager: SOTAManager,
    validation_payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            line = _report_jsonl_line(
                result=result,
                tool_name=tool_name,
                protocol_id=protocol_id,
                run_dir=run_dir,
                model_dir=model_dir,
                sota_manager=sota_manager,
                validation_payload=validation_payload,
            )
            handle.write(json.dumps(_to_strict_jsonable(line), ensure_ascii=False, allow_nan=False) + "\n")
    logger.info("Wrote dataset-metric report", path=str(path), num_entries=len(results))


def _payload_result_row(result: dict[str, Any]) -> dict[str, Any]:
    pipeline_description = result.get("pipeline_description") or {}
    nodes = pipeline_description.get("nodes") or []
    return {
        "schema": "sure.eval.payload.dataset_metric.v2",
        "dataset": result["dataset"],
        "task": result["task"],
        "language": result["language"],
        "metric": result["metric"],
        "result": _compact_metric_result(result),
        "baseline_dataset": result.get("baseline_dataset"),
        "rps": result.get("rps"),
        "evaluation_context": result.get("evaluation_context") or {},
        "pipeline": {
            "pipeline_id": result.get("pipeline_id"),
            "description_path": _display_path(result.get("pipeline_description_path")),
            "report_path": _display_path(result.get("metric_report_path")),
            "nodes": nodes,
            "conversion_steps": pipeline_description.get("conversion_steps") or [],
        },
        "artifacts": {
            "metric_artifact_dir": _display_path(result.get("metric_artifact_dir")),
            "sample_report": _display_path(result.get("sample_report_path")),
        },
        "inputs": {
            "jsonl_path": _display_path(result.get("jsonl_path")),
            "prediction_path": _display_path(result.get("prediction_path")),
        },
    }


def _build_evaluation_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "sure.eval.payload.v2",
        "result_granularity": "dataset_metric",
        "sample_level_results": "excluded_from_final_payload",
        "sample_report_layout": "sample_reports/<dataset>/<metric_slug>.jsonl",
        "results": [_payload_result_row(result) for result in results],
    }


def _legacy_result_from_payload_row(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a v2 payload row back to the internal row shape used by writers."""

    if "result" not in result:
        return dict(result)

    metric = str(result.get("metric"))
    pipeline = result.get("pipeline") or {}
    artifacts = result.get("artifacts") or {}
    inputs = result.get("inputs") or {}
    metric_result = result.get("result") or {}
    score = metric_result.get("score")
    if score is None:
        score_key = metric_result.get("score_key")
        score = metric_result.get(score_key) if score_key else None
    if score is None:
        raise ValueError(f"merged v2 payload row is missing result.score for metric {metric}")

    internal = {
        "dataset": result.get("dataset"),
        "jsonl_path": inputs.get("jsonl_path") or result.get("jsonl_path"),
        "prediction_path": inputs.get("prediction_path") or result.get("prediction_path"),
        "task": result.get("task"),
        "language": result.get("language"),
        "metric": metric,
        "baseline_dataset": result.get("baseline_dataset"),
        "score": score,
        "rps": result.get("rps"),
        "rps_is_unbounded": isinstance(result.get("rps"), float) and not math.isfinite(result["rps"]),
        "num_samples": metric_result.get("num_samples"),
        "evaluation_context": result.get("evaluation_context") or {},
        "pipeline_id": pipeline.get("pipeline_id"),
        "metric_artifact_dir": artifacts.get("metric_artifact_dir"),
        "metric_report_path": pipeline.get("report_path"),
        "pipeline_description_path": pipeline.get("description_path"),
        "pipeline_description": {},
        "sample_report_path": artifacts.get("sample_report"),
        "details": dict(metric_result),
    }
    score_key = metric_result.get("score_key")
    if score_key and score_key not in internal["details"]:
        internal["details"][score_key] = score
    return internal


def _write_sample_reports(sample_reports_dir: Path, results: list[dict[str, Any]]) -> None:
    from sure_eval.evaluation.nodes.scoring.wenet_wer.wenet_compute_cer import Calculator, characterize

    for result in results:
        dataset_name = result["dataset"]
        metric = str(result["metric"])
        output_path = sample_reports_dir / dataset_name / f"{_metric_slug(metric)}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path = _localize_path(result["jsonl_path"])
        prediction_path = _localize_path(result["prediction_path"])
        samples = load_jsonl(jsonl_path)
        predictions = load_prediction_map(prediction_path)
        with output_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                key = str(sample.get("key", ""))
                pred = predictions.get(key, "")
                row: dict[str, Any] = {"key": key, "metric": metric, "prediction": pred}
                if result["task"] == "ASR":
                    ref = str(sample.get("target", ""))
                    calc = Calculator()
                    if metric == "cer":
                        ref_tokens = characterize(ref)
                        pred_tokens = characterize(pred)
                    else:
                        ref_tokens = ref.upper().split()
                        pred_tokens = pred.upper().split()
                    detail = calc.calculate(ref_tokens, pred_tokens)
                    total = detail["all"]
                    errors = detail["sub"] + detail["ins"] + detail["del"]
                    row.update(
                        {
                            "reference": ref,
                            "score": round(errors / total, 6) if total > 0 else 0.0,
                            "counts": {
                                "all": detail["all"],
                                "cor": detail["cor"],
                                "sub": detail["sub"],
                                "ins": detail["ins"],
                                "del": detail["del"],
                            },
                        }
                    )
                elif result["task"] == "TTS":
                    row.update(
                        {
                            "generated_audio": pred,
                            "reference_text": sample.get("reference_text") or sample.get("target"),
                            "reference_audio": sample.get("reference_audio") or sample.get("path"),
                        }
                    )
                else:
                    row.update({"reference": sample.get("target")})
                handle.write(json.dumps(_to_strict_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
        result["sample_report_path"] = str(output_path)
        logger.info("Wrote sample report", dataset=dataset_name, metric=metric, path=str(output_path))


def _write_results_dir(
    results_dir: Path,
    *,
    source_run_dir: Path | None,
    report_jsonl: Path | None,
    protocol_yaml: Path | None,
    results: list[dict[str, Any]],
) -> None:
    """Write a compatibility mirror under results/<model>/<protocol>."""

    results_dir.mkdir(parents=True, exist_ok=True)
    if report_jsonl and report_jsonl.exists():
        shutil.copy2(report_jsonl, results_dir / "report.jsonl")
    if protocol_yaml and protocol_yaml.exists():
        shutil.copy2(protocol_yaml, results_dir / "protocol.yaml")
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        prediction_path = Path(result["prediction_path"])
        if prediction_path.exists():
            shutil.copy2(prediction_path, pred_dir / prediction_path.name)
        structured_prediction_path = prediction_path.with_suffix(".jsonl")
        if structured_prediction_path.exists():
            shutil.copy2(structured_prediction_path, pred_dir / structured_prediction_path.name)
    if source_run_dir and (source_run_dir / "report_snapshot.md").exists():
            shutil.copy2(source_run_dir / "report_snapshot.md", results_dir / "report_snapshot.md")


def merge_payload_results(payload_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Merge result rows from one or more evaluate_predictions payloads."""

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for payload_path in payload_paths:
        path = Path(payload_path)
        payload = _read_json(path)
        for result in payload.get("results") or []:
            result = _legacy_result_from_payload_row(result)
            key = (str(result.get("dataset")), str(result.get("metric")))
            if key in seen:
                raise ValueError(f"duplicate dataset/metric result in merged payloads: {key[0]} {key[1]}")
            seen.add(key)
            merged_result = dict(result)
            metric_artifact_dir = merged_result.get("metric_artifact_dir")
            if metric_artifact_dir:
                artifact_dir = Path(str(metric_artifact_dir))
                description_path = artifact_dir / "pipeline_description.json"
                report_path = artifact_dir / "report.json"
                merged_result.setdefault("pipeline_description_path", str(description_path))
                merged_result.setdefault("metric_report_path", str(report_path))
                if not merged_result.get("pipeline_description"):
                    merged_result["pipeline_description"] = _read_json(description_path)
                if "pipeline_id" not in merged_result and isinstance(merged_result.get("pipeline_description"), dict):
                    merged_result["pipeline_id"] = merged_result["pipeline_description"].get("pipeline_id")
            results.append(merged_result)
    return results


def _copy_metric_artifacts_to_run_dir(results: list[dict[str, Any]], run_dir: Path | None) -> None:
    if run_dir is None:
        return
    for result in results:
        source_value = result.get("metric_artifact_dir")
        if not source_value:
            continue
        source_dir = Path(str(source_value))
        if not source_dir.exists():
            continue
        metric_slug = _metric_slug(str(result["metric"]))
        destination_dir = run_dir / "metrics" / str(result["dataset"]) / metric_slug
        if source_dir.resolve() == destination_dir.resolve():
            result["metric_artifact_dir"] = str(destination_dir)
            result["metric_report_path"] = str(destination_dir / "report.json")
            result["pipeline_description_path"] = str(destination_dir / "pipeline_description.json")
            if not result.get("pipeline_description"):
                result["pipeline_description"] = _read_json(destination_dir / "pipeline_description.json")
            if result["pipeline_description"]:
                result["pipeline_id"] = result["pipeline_description"].get("pipeline_id", result.get("pipeline_id"))
            continue
        destination_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("report.json", "pipeline_description.json"):
            source_file = source_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, destination_dir / filename)
        result["metric_artifact_dir"] = str(destination_dir)
        result["metric_report_path"] = str(destination_dir / "report.json")
        result["pipeline_description_path"] = str(destination_dir / "pipeline_description.json")
        result["pipeline_description"] = _read_json(destination_dir / "pipeline_description.json")
        if result["pipeline_description"]:
            result["pipeline_id"] = result["pipeline_description"].get("pipeline_id", result.get("pipeline_id"))


def write_standard_evaluation_artifacts(
    *,
    results: list[dict[str, Any]],
    run_dir: Path | None,
    results_dir: Path | None,
    tool_name: str,
    protocol_id: str,
    model_dir: Path | None,
    validation_payload: dict[str, Any],
    output_path: Path | None,
    sample_reports_dir: Path | None = None,
    report_jsonl_path: Path | None = None,
    protocol_output_path: Path | None = None,
) -> None:
    """Write canonical main-flow evaluation artifacts for result rows."""

    if run_dir:
        sample_reports_dir = sample_reports_dir or run_dir / "sample_reports"
        report_jsonl_path = report_jsonl_path or run_dir / "report.jsonl"
        protocol_output_path = protocol_output_path or run_dir / "protocol.yaml"

    _copy_metric_artifacts_to_run_dir(results, run_dir)

    if sample_reports_dir:
        _write_sample_reports(sample_reports_dir, results)

    payload = _to_strict_jsonable(_build_evaluation_payload(results))
    output = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    print(output)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        logger.info("Wrote evaluation payload", path=str(output_path))

    sota_manager = SOTAManager()
    if protocol_output_path:
        _write_protocol_yaml(
            protocol_output_path,
            run_dir=run_dir,
            tool_name=tool_name,
            protocol_id=protocol_id,
            model_dir=model_dir,
            results=results,
        )

    if report_jsonl_path:
        _write_report_jsonl(
            report_jsonl_path,
            results=results,
            tool_name=tool_name,
            protocol_id=protocol_id,
            run_dir=run_dir,
            model_dir=model_dir,
            sota_manager=sota_manager,
            validation_payload=validation_payload,
        )

    if results_dir:
        _write_results_dir(
            results_dir=results_dir,
            source_run_dir=run_dir,
            report_jsonl=report_jsonl_path,
            protocol_yaml=protocol_output_path,
            results=results,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic prediction files")
    parser.add_argument("--dataset", nargs="+", required=True, help="Dataset names to evaluate")
    parser.add_argument("--pred-dir", type=str, help="Directory containing <dataset>.txt prediction files")
    parser.add_argument("--pred", action="append", nargs=2, metavar=("DATASET", "FILE"), help="Explicit dataset-to-prediction mapping")
    parser.add_argument("--tool-name", type=str, help="Optional tool name to record in evaluation history")
    parser.add_argument("--record", action="store_true", help="Record results in the evaluation database")
    parser.add_argument("--config", type=str, help="Config path")
    parser.add_argument("--output", type=str, help="Optional JSON output path")
    parser.add_argument("--results-dir", type=str, help="Results output directory (e.g., results/asr_qwen3/strict_core)")
    parser.add_argument("--protocol-id", type=str, default="strict_core", help="Inference protocol ID")
    parser.add_argument("--model-dir", type=str, help="Model directory to extract protocol.yaml from config.yaml")
    parser.add_argument("--run-dir", type=str, help="Main-flow run directory for run-local report artifacts")
    parser.add_argument("--report-jsonl", type=str, help="Optional run-local report.jsonl output path")
    parser.add_argument("--protocol-output", type=str, help="Optional run-local protocol.yaml output path")
    parser.add_argument("--metrics-dir", type=str, help="Optional metrics artifact directory")
    parser.add_argument("--sample-reports-dir", type=str, help="Optional sample report directory")
    parser.add_argument("--validation-payload", type=str, help="Optional validation_payload.json path to fold into report.jsonl")
    parser.add_argument("--device", default="cuda", help="Device passed to TTS metric providers, for example cuda:0 or cpu")
    parser.add_argument(
        "--merge-payload",
        action="append",
        help="Merge an existing evaluation_payload.json instead of running metrics. Repeat for segmented TTS/VC evaluation.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        help="Metric to evaluate. Repeat for multiple metrics per dataset. Defaults to SOTA baseline or task default.",
    )
    args = parser.parse_args()

    dataset_manager, rps_manager = _build_dataset_and_rps_managers(args.config)
    sota_manager = SOTAManager()

    explicit_preds = {dataset_manager.normalize_dataset_name(name): Path(path) for name, path in (args.pred or [])}
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    model_dir = Path(args.model_dir).resolve() if args.model_dir else None
    metrics_root = Path(args.metrics_dir) if args.metrics_dir else (run_dir / "metrics" if run_dir else None)
    sample_reports_dir = Path(args.sample_reports_dir) if args.sample_reports_dir else (run_dir / "sample_reports" if run_dir else None)
    report_jsonl_path = Path(args.report_jsonl) if args.report_jsonl else (run_dir / "report.jsonl" if run_dir else None)
    protocol_output_path = Path(args.protocol_output) if args.protocol_output else (run_dir / "protocol.yaml" if run_dir else None)
    validation_payload = _read_json(Path(args.validation_payload)) if args.validation_payload else (
        _read_json(run_dir / "validation_payload.json") if run_dir else {}
    )

    if args.merge_payload:
        results = merge_payload_results([Path(path) for path in args.merge_payload])
        write_standard_evaluation_artifacts(
            results=results,
            run_dir=run_dir,
            results_dir=Path(args.results_dir) if args.results_dir else None,
            tool_name=args.tool_name or "unknown",
            protocol_id=args.protocol_id,
            model_dir=model_dir,
            validation_payload=validation_payload,
            output_path=Path(args.output) if args.output else None,
            sample_reports_dir=sample_reports_dir,
            report_jsonl_path=report_jsonl_path,
            protocol_output_path=protocol_output_path,
        )
        return 0

    metric_overrides: list[str | None]
    if args.metric:
        metric_overrides = []
        for metric_name in args.metric:
            metric_name = metric_name.strip()
            if metric_name and metric_name not in metric_overrides:
                metric_overrides.append(metric_name)
        if not metric_overrides:
            raise ValueError("--metric was provided but no non-empty metric names were found")
    else:
        metric_overrides = [None]

    results: list[dict[str, Any]] = []
    for requested_dataset in dataset_manager.expand_dataset_names(args.dataset):
        canonical_name = dataset_manager.normalize_dataset_name(requested_dataset)
        task, language = _dataset_task_language(dataset_manager, canonical_name)
        prediction_path = explicit_preds.get(canonical_name)
        if prediction_path is None:
            if pred_dir is None:
                raise ValueError(f"No prediction file provided for dataset: {canonical_name}")
            prediction_path = pred_dir / f"{canonical_name}.txt"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

        applicable_metric_overrides = [
            metric_override
            for metric_override in metric_overrides
            if _metric_applies_to_task_language(metric_override, task, language)
        ]
        if not applicable_metric_overrides:
            requested = ", ".join(metric for metric in metric_overrides if metric) or "default"
            raise ValueError(
                f"No requested metric applies to dataset {canonical_name} "
                f"(task={task}, language={language}, requested={requested})"
            )

        for metric_override in applicable_metric_overrides:
            result = evaluate_prediction_file(
                dataset_manager,
                sota_manager,
                canonical_name,
                prediction_path,
                metric_override=metric_override,
                metrics_root=metrics_root,
                device=args.device,
            )
            results.append(result)

            if args.record:
                if not args.tool_name:
                    raise ValueError("--record requires --tool-name")
                if isinstance(result.get("rps"), dict):
                    logger.info(
                        "Skipping RPS database record for non-comparable metric",
                        dataset=result["dataset"],
                        metric=result["metric"],
                        rps_status=result["rps"],
                    )
                elif getattr(rps_manager, "database", None) is not None:
                    from sure_eval.evaluation.rps import EvaluationRecord

                    record = EvaluationRecord(
                        tool_name=args.tool_name,
                        model_name=None,
                        dataset=result["dataset"],
                        score=result["score"],
                        metric=result["metric"],
                        rps=result["rps"],
                        metadata={
                            "num_samples": result["num_samples"],
                            "prediction_path": result["prediction_path"],
                            "baseline_dataset": result.get("baseline_dataset"),
                            "details": result["details"],
                        },
                    )
                    rps_manager.database.add_record(record)
                else:
                    logger.info(
                        "Skipping RPS database record because minimal evaluation runtime is active",
                        dataset=result["dataset"],
                        metric=result["metric"],
                    )

    write_standard_evaluation_artifacts(
        results=results,
        run_dir=run_dir,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        tool_name=args.tool_name or "unknown",
        protocol_id=args.protocol_id,
        model_dir=model_dir,
        validation_payload=validation_payload,
        output_path=Path(args.output) if args.output else None,
        sample_reports_dir=sample_reports_dir,
        report_jsonl_path=report_jsonl_path,
        protocol_output_path=protocol_output_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
