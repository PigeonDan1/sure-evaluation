"""Provider implementations for TTS semantic error-rate metrics."""

from __future__ import annotations

import json
import importlib.metadata
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


QWEN3_ASR_1_7B_NODE_ID = "transcription/qwen3_asr_1_7b"
QWEN3_ASR_1_7B_NODE_VERSION = "v1"
QWEN3_ASR_1_7B_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN3_ASR_RUNTIME_PACKAGE = "qwen-asr"
QWEN3_ASR_DECLARED_PACKAGE_VERSION = "0.0.6"
QWEN3_ASR_CHECKPOINT_ENV = "QWEN3_ASR_1_7B_CHECKPOINT"
QWEN3_ASR_DEFAULT_MAX_INFERENCE_BATCH_SIZE = 32
QWEN3_ASR_DEFAULT_MAX_NEW_TOKENS = 256
QWEN3_ASR_RUNTIME_SAMPLE_RATE_HZ = 16000


def qwen3_asr_language_hint(language: str) -> str | None:
    """Return the Qwen3-ASR language prompt used for routed TTS evaluation."""

    normalized = str(language).strip().lower()
    if normalized.startswith(("zh", "cmn")):
        return "Chinese"
    if normalized.startswith("yue"):
        return "Cantonese"
    if normalized.startswith("en"):
        return "English"
    return None


def qwen3_asr_trace_details(
    *,
    audio_path: str,
    language: str,
    role: str,
    transcript: str,
    runner: Any | None = None,
    detected_language: str | None = None,
    language_hint: str | None = None,
) -> dict[str, Any]:
    """Build stable trace metadata for the Qwen3-ASR transcription node."""

    return {
        "audio_path": audio_path,
        "language": language,
        "role": role,
        "transcript": transcript,
        "model_id": getattr(runner, "model_id", QWEN3_ASR_1_7B_MODEL_ID),
        "resolved_model_id": getattr(runner, "resolved_model_id", None),
        "runtime_package": QWEN3_ASR_RUNTIME_PACKAGE,
        "runtime_package_version": getattr(
            runner,
            "runtime_package_version",
            _installed_package_version(QWEN3_ASR_RUNTIME_PACKAGE)
            or QWEN3_ASR_DECLARED_PACKAGE_VERSION,
        ),
        "backend": getattr(runner, "backend", "transformers"),
        "audio_input_mode": "path",
        "audio_frontend_policy": "runtime_managed",
        "resample_policy": "qwen_asr_runtime_managed",
        "runtime_audio_normalizer": "qwen_asr.inference.utils.normalize_audio_input",
        "runtime_normalized_sample_rate_hz": QWEN3_ASR_RUNTIME_SAMPLE_RATE_HZ,
        "runtime_normalized_channels": 1,
        "runtime_audio_dtype": "float32",
        "external_frontend_node": None,
        "dtype": getattr(runner, "dtype_name", "auto_bfloat16_cuda_else_float32"),
        "device_map": getattr(runner, "device_map", None),
        "max_inference_batch_size": getattr(
            runner,
            "max_inference_batch_size",
            QWEN3_ASR_DEFAULT_MAX_INFERENCE_BATCH_SIZE,
        ),
        "max_new_tokens": getattr(
            runner,
            "max_new_tokens",
            QWEN3_ASR_DEFAULT_MAX_NEW_TOKENS,
        ),
        "language_hint": language_hint or qwen3_asr_language_hint(language),
        "detected_language": detected_language,
        "timestamps_enabled": False,
        "forced_aligner": None,
    }


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
            payload_internal_stages = payload.get("internal_stages")
            internal_stages = (
                tuple(str(item) for item in payload_internal_stages)
                if isinstance(payload_internal_stages, list)
                else ("audio_decode", "asr_inference", "text_extraction")
            )
            results.append(
                (
                    str(payload.get("transcript", "")),
                    PipelineNodeResult(
                        stage="transcription",
                        node_id=str(payload.get("node_id", self.node_id)),
                        version=str(payload.get("version", "v1")),
                        details=trace_details,
                        internal_stages=internal_stages,
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


class Qwen3ASR17BTranscriber:
    """Qwen3-ASR-1.7B transcriber for explicit TTS semantic routes."""

    model_id = QWEN3_ASR_1_7B_MODEL_ID
    backend = "transformers"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
        *,
        max_inference_batch_size: int = QWEN3_ASR_DEFAULT_MAX_INFERENCE_BATCH_SIZE,
        max_new_tokens: int = QWEN3_ASR_DEFAULT_MAX_NEW_TOKENS,
        dtype: str = "auto",
        device_map: str | None = None,
    ) -> None:
        self.model_id = model_id or self.model_id
        self.device = device
        self.cache_dir = cache_dir
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.device_map = device_map or self._normalize_device_map(device)
        self.dtype_name = self._dtype_name(dtype=dtype, device_map=self.device_map)
        self.runtime_package_version = (
            _installed_package_version(QWEN3_ASR_RUNTIME_PACKAGE)
            or QWEN3_ASR_DECLARED_PACKAGE_VERSION
        )
        self.resolved_model_id: str | None = None
        self.last_detected_language: str | None = None
        self.last_language_hint: str | None = None
        self._model: Any | None = None

    def _resolved_model_id(self) -> str:
        explicit_checkpoint = os.environ.get(QWEN3_ASR_CHECKPOINT_ENV)
        if explicit_checkpoint:
            return str(Path(explicit_checkpoint).expanduser())
        return self.model_id

    def _load(self) -> Any:
        if self._model is None:
            configure_model_cache(self.cache_dir)
            if self.cache_dir is not None:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            install_deepspeed_stub()

            import torch
            from qwen_asr import Qwen3ASRModel

            self.resolved_model_id = self._resolved_model_id()
            self._model = Qwen3ASRModel.from_pretrained(
                self.resolved_model_id,
                dtype=self._torch_dtype(torch),
                device_map=self.device_map,
                max_inference_batch_size=self.max_inference_batch_size,
                max_new_tokens=self.max_new_tokens,
            )
        return self._model

    def transcribe(self, audio_path: str, *, language: str = "en") -> str:
        language_hint = qwen3_asr_language_hint(language)
        result = self._transcribe_raw(audio_path, language_hint=language_hint)
        transcript, detected_language = _qwen3_asr_result_text_and_language(result)
        self.last_detected_language = detected_language
        self.last_language_hint = language_hint
        return transcript

    def transcribe_batch(
        self,
        audio_paths: list[str],
        *,
        language: str = "en",
        role: str = "prediction_audio",
    ) -> list[tuple[str, PipelineNodeResult]]:
        if not audio_paths:
            return []

        language_hint = qwen3_asr_language_hint(language)
        raw_results = self._transcribe_raw(audio_paths, language_hint=language_hint)
        if not isinstance(raw_results, list):
            raw_results = [raw_results]
        if len(raw_results) != len(audio_paths):
            raise RuntimeError(
                f"{QWEN3_ASR_1_7B_NODE_ID} returned {len(raw_results)} transcript(s) "
                f"for {len(audio_paths)} input(s)"
            )

        results: list[tuple[str, PipelineNodeResult]] = []
        for audio_path, raw_result in zip(audio_paths, raw_results, strict=True):
            transcript, detected_language = _qwen3_asr_result_text_and_language(raw_result)
            trace_details = qwen3_asr_trace_details(
                audio_path=audio_path,
                language=language,
                role=role,
                transcript=transcript,
                runner=self,
                detected_language=detected_language,
                language_hint=language_hint,
            )
            results.append(
                (
                    transcript,
                    PipelineNodeResult(
                        stage="transcription",
                        node_id=QWEN3_ASR_1_7B_NODE_ID,
                        version=QWEN3_ASR_1_7B_NODE_VERSION,
                        details=trace_details,
                        internal_stages=(
                            "runtime_managed_audio_frontend",
                            "asr_inference",
                            "text_extraction",
                        ),
                    ),
                )
            )
        return results

    def _transcribe_raw(self, audio: str | list[str], *, language_hint: str | None) -> Any:
        if language_hint:
            return self._load().transcribe(audio=audio, language=language_hint)
        return self._load().transcribe(audio=audio)

    def _torch_dtype(self, torch_module: Any) -> Any:
        dtype = str(self.dtype).strip().lower()
        if dtype in {"auto", ""}:
            return torch_module.float32 if self.device_map == "cpu" else torch_module.bfloat16
        if dtype in {"bfloat16", "bf16", "torch.bfloat16"}:
            return torch_module.bfloat16
        if dtype in {"float16", "fp16", "torch.float16"}:
            return torch_module.float16
        if dtype in {"float32", "fp32", "torch.float32"}:
            return torch_module.float32
        raise ValueError(f"Unsupported Qwen3-ASR dtype: {self.dtype!r}")

    @staticmethod
    def _normalize_device_map(device: str) -> str:
        normalized = str(device).strip().lower()
        if normalized in {"", "cuda"}:
            return "cuda:0"
        if normalized in {"cpu", "mps"}:
            return normalized
        if normalized.startswith("cuda:"):
            return normalized
        if normalized.isdigit():
            return f"cuda:{normalized}"
        return str(device)

    @staticmethod
    def _dtype_name(*, dtype: str, device_map: str) -> str:
        normalized = str(dtype).strip().lower()
        if normalized in {"", "auto"}:
            return "float32" if device_map == "cpu" else "bfloat16"
        return normalized.removeprefix("torch.")


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


def _qwen3_asr_result_text_and_language(result: Any) -> tuple[str, str | None]:
    if isinstance(result, (list, tuple)):
        if not result:
            return "", None
        return _qwen3_asr_result_text_and_language(result[0])
    if isinstance(result, dict):
        text = str(result.get("text", ""))
        language = result.get("language")
        return text, str(language) if language else None
    text = str(getattr(result, "text", ""))
    language = getattr(result, "language", None)
    if not text:
        text = str(result)
    return text, str(language) if language else None


def _installed_package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
