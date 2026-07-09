"""Provider implementations for TTS semantic error-rate metrics."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sure_eval.compat.deepspeed_stub import install_deepspeed_stub
from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.common.node_local_python import (
    build_node_local_env,
    resolve_node_local_python,
)


class Transcriber(Protocol):
    """Protocol for audio-to-text runners."""

    def transcribe(self, audio_path: str, *, language: str = "en") -> str:
        """Transcribe one audio file."""
        ...


class BatchTranscriber(Transcriber, Protocol):
    """Protocol for runners that can transcribe a batch with one model load."""

    def transcribe_batch(
        self,
        audio_paths: list[str],
        *,
        language: str = "en",
        role: str = "prediction_audio",
    ) -> list[tuple[str, PipelineNodeResult]]:
        """Transcribe multiple audio files and return node traces."""
        ...


@dataclass(frozen=True)
class StaticTranscriber:
    """Test and adapter transcriber that returns a fixed transcript."""

    transcript: str

    def transcribe(self, audio_path: str, *, language: str = "en") -> str:
        return self.transcript


@dataclass(frozen=True)
class NodeLocalTranscriber:
    """Transcriber that calls a transcription node through its local uv env."""

    node_id: str
    node_dir: Path
    device: str = "cuda"

    def transcribe(self, audio_path: str, *, language: str = "en") -> str:
        completed = self._run_node_command(
            [
                "--audio-path",
                audio_path,
                "--language",
                language,
                "--device",
                self.device,
                "--json",
            ]
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{self.node_id} did not return JSON: {completed.stdout[:500]}") from exc
        return str(payload.get("transcript", ""))

    def transcribe_batch(
        self,
        audio_paths: list[str],
        *,
        language: str = "en",
        role: str = "prediction_audio",
    ) -> list[tuple[str, PipelineNodeResult]]:
        if not audio_paths:
            return []

        chunk_size = _transcribe_batch_size(self.node_id)
        if chunk_size and len(audio_paths) > chunk_size:
            results: list[tuple[str, PipelineNodeResult]] = []
            for chunk in _chunk_audio_paths(audio_paths, chunk_size):
                results.extend(self._transcribe_batch_once(chunk, language=language, role=role))
            return results

        return self._transcribe_batch_once(audio_paths, language=language, role=role)

    def _transcribe_batch_once(
        self,
        audio_paths: list[str],
        *,
        language: str,
        role: str,
    ) -> list[tuple[str, PipelineNodeResult]]:

        input_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        try:
            input_path = Path(input_file.name)
            for audio_path in audio_paths:
                input_file.write(
                    json.dumps(
                        {
                            "audio_path": audio_path,
                            "language": language,
                            "role": role,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            input_file.close()
            completed = self._run_node_command(
                [
                    "--input-jsonl",
                    str(input_path),
                    "--device",
                    self.device,
                    "--json",
                ]
            )
        finally:
            input_file.close()
            Path(input_file.name).unlink(missing_ok=True)

        results: list[tuple[str, PipelineNodeResult]] = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{self.node_id} returned invalid JSONL row: {line[:500]}") from exc
            trace_details = payload.get("trace")
            if not isinstance(trace_details, dict):
                trace_details = {
                    "audio_path": payload.get("audio_path", ""),
                    "language": payload.get("language", language),
                    "role": role,
                    "transcript": payload.get("transcript", ""),
                }
            results.append(
                (
                    str(payload.get("transcript", "")),
                    PipelineNodeResult(
                        stage="transcription",
                        node_id=str(payload.get("node_id", self.node_id)),
                        version=str(payload.get("version", "v1")),
                        details=trace_details,
                        internal_stages=("audio_decode", "asr_inference", "text_extraction"),
                    ),
                )
            )
        if len(results) != len(audio_paths):
            raise RuntimeError(f"{self.node_id} returned {len(results)} transcript(s) for {len(audio_paths)} input(s)")
        return results

    def _run_node_command(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        python_runtime = resolve_node_local_python(self.node_dir, self.node_id)
        command = [
            *python_runtime.command_prefix,
            "-m",
            f"sure_eval.evaluation.nodes.transcription.{self.node_id.split('/', 1)[1]}.node",
            *args,
        ]
        repo_root = self.node_dir.parents[5]
        env = build_node_local_env(
            repo_src=repo_root / "src",
            extra_pythonpath=python_runtime.extra_pythonpath,
            inherit_pythonpath=python_runtime.inherit_pythonpath,
        )
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.node_id} transcription failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return completed


def configure_model_cache(cache_dir: str | Path | None) -> None:
    """Point common model download libraries at the shared TTS metric cache."""
    if cache_dir is None:
        return
    cache_path = Path(cache_dir)
    os.environ["HF_HOME"] = str(cache_path / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_path / "huggingface" / "hub")
    os.environ["MODELSCOPE_CACHE"] = str(cache_path / "modelscope")
    os.environ["TORCH_HOME"] = str(cache_path / "torch")


def normalize_transformers_device(device: str | int | None) -> str | int | None:
    """Convert common SURE device strings to Transformers pipeline devices."""
    if device is None or isinstance(device, int):
        return device
    normalized = str(device).strip().lower()
    if normalized in {"cpu", "-1"}:
        return -1
    if normalized == "cuda":
        return 0
    if normalized.startswith("cuda:"):
        return int(normalized.split(":", 1)[1])
    try:
        return int(normalized)
    except ValueError:
        return device


def _transcribe_batch_size(node_id: str) -> int:
    normalized_node_id = "".join(ch if ch.isalnum() else "_" for ch in node_id.upper()).strip("_")
    env_names = (
        f"SURE_EVAL_NODE_LOCAL_TRANSCRIBE_BATCH_SIZE_{normalized_node_id}",
        f"SURE_EVAL_TRANSCRIPTION_BATCH_SIZE_{normalized_node_id}",
        "SURE_EVAL_NODE_LOCAL_TRANSCRIBE_BATCH_SIZE",
        "SURE_EVAL_TRANSCRIPTION_BATCH_SIZE",
    )
    env_name = env_names[-1]
    raw_value = "0"
    for candidate in env_names:
        value = os.environ.get(candidate)
        if value is not None:
            env_name = candidate
            raw_value = value.strip()
            break
    if not raw_value:
        return 0
    try:
        batch_size = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    if batch_size < 0:
        raise RuntimeError(f"{env_name} must be non-negative, got {batch_size}")
    return batch_size


def _chunk_audio_paths(audio_paths: list[str], chunk_size: int) -> list[list[str]]:
    return [audio_paths[index : index + chunk_size] for index in range(0, len(audio_paths), chunk_size)]


class WhisperLargeV3Transcriber:
    """Whisper-large-v3 English transcriber used by Seed-TTS-Eval style WER."""

    model_id = "openai/whisper-large-v3"

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id or self.model_id
        self.device = device
        self.cache_dir = cache_dir
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        if self._pipeline is None:
            configure_model_cache(self.cache_dir)
            if self.cache_dir is not None:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            # Some base runtime containers contain a broken optional deepspeed install.
            # Transformers can import it while importing pipelines; Whisper
            # inference does not need deepspeed.
            install_deepspeed_stub()
            from transformers import pipeline

            kwargs: dict[str, Any] = {
                "task": "automatic-speech-recognition",
            }
            if self.cache_dir is not None:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

                model_cache = str(Path(self.cache_dir) / "huggingface" / "hub")
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    cache_dir=model_cache,
                    local_files_only=True,
                )
                processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=model_cache,
                    local_files_only=True,
                )
                kwargs["model"] = model
                kwargs["tokenizer"] = processor.tokenizer
                kwargs["feature_extractor"] = processor.feature_extractor
            else:
                kwargs["model"] = self.model_id
            if self.device is not None:
                kwargs["device"] = normalize_transformers_device(self.device)
            self._pipeline = pipeline(**kwargs)
        return self._pipeline

    def transcribe(self, audio_path: str, *, language: str = "en") -> str:
        result = self._load()(audio_path, generate_kwargs={"language": "english", "task": "transcribe"})
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return str(result)


class ParaformerZHTranscriber:
    """Paraformer Chinese transcriber used by Seed-TTS-Eval style CER."""

    model_id = "paraformer-zh"
    local_modelscope_id = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id or self.model_id
        self.device = device
        self.cache_dir = cache_dir
        self._model: Any | None = None

    def _resolved_model_id(self) -> str:
        if self.cache_dir is None:
            return self.model_id
        local_model_dir = Path(self.cache_dir) / "modelscope" / "models" / Path(self.local_modelscope_id)
        if (local_model_dir / "configuration.json").exists():
            return str(local_model_dir)
        return self.model_id

    def _load(self) -> Any:
        if self._model is None:
            configure_model_cache(self.cache_dir)
            from funasr import AutoModel

            model_id = self._resolved_model_id()
            try:
                self._model = AutoModel(model=model_id, device=self.device, disable_update=True)
            except TypeError as exc:
                if "disable_update" not in str(exc):
                    raise
                self._model = AutoModel(model=model_id, device=self.device)
        return self._model

    def transcribe(self, audio_path: str, *, language: str = "zh") -> str:
        result = self._load().generate(input=audio_path, batch_size_s=300)
        if isinstance(result, list) and result:
            return str(result[0].get("text", ""))
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return str(result)


class TTSSemanticErrorRateProvider:
    """Score TTS intelligibility by transcribing audio and applying SURE WER/CER."""

    def __init__(self, transcriber: Transcriber) -> None:
        self.transcriber = transcriber

    def __call__(
        self,
        prediction: str,
        reference: str,
        *,
        language: str = "en",
        metric: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from sure_eval.evaluation.tasks.tts.metrics import CERMetric, WERMetric

        transcript = self.transcriber.transcribe(prediction, language=language)
        metric_name = metric or ("cer" if language == "zh" else "wer")
        scorer = CERMetric() if metric_name in {"cer", "tts_cer"} else WERMetric()
        result = scorer.calculate(transcript, reference, language=language, **kwargs)
        score_key = "cer" if isinstance(scorer, CERMetric) else "wer"
        return {
            score_key: float(result.score),
            "score": float(result.score),
            "transcript": transcript,
            "reference_text": reference,
            "audio_path": str(Path(prediction)),
            "sure_result": result.details.get("sure_result", {}),
        }
