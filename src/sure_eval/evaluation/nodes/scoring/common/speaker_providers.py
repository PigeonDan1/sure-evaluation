"""Provider implementations for speaker-similarity metrics."""

from __future__ import annotations

import os
import tempfile
from contextlib import nullcontext
from typing import Callable, Sequence
from pathlib import Path
from typing import Any

from sure_eval.compat.deepspeed_stub import install_deepspeed_stub
from sure_eval.evaluation.nodes.transcription.common.providers import (
    configure_model_cache,
    normalize_transformers_device,
)


Embedding = Sequence[float]
EmbeddingProvider = Callable[[str], Embedding]


def _writable_cache_subdir(cache_path: Path, *parts: str) -> Path:
    if os.access(cache_path, os.W_OK):
        return cache_path.joinpath(*parts)
    return Path("/tmp") / "sure-eval-metric-cache" / cache_path.name / Path(*parts)


def normalize_modelscope_device(device: str | int | None) -> str | None:
    """Convert common SURE device strings to ModelScope pipeline devices."""
    if device is None:
        return None
    if isinstance(device, int):
        return "cpu" if device < 0 else f"cuda:{device}"
    normalized = str(device).strip().lower()
    if normalized in {"cpu", "-1"}:
        return "cpu"
    if normalized == "cuda":
        return "cuda:0"
    if normalized.startswith("cuda:"):
        return normalized
    try:
        device_index = int(normalized)
    except ValueError:
        return str(device)
    return "cpu" if device_index < 0 else f"cuda:{device_index}"


def cosine_similarity(left: Embedding, right: Embedding) -> float:
    """Return cosine similarity between two embedding vectors."""
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values):
        raise ValueError("embedding vectors must have the same length")

    dot = sum(l * r for l, r in zip(left_values, right_values, strict=True))
    left_norm = sum(value * value for value in left_values) ** 0.5
    right_norm = sum(value * value for value in right_values) ** 0.5
    if left_norm == 0 or right_norm == 0:
        raise ValueError("embedding vectors must be non-zero")
    return dot / (left_norm * right_norm)


def _torch_inference_mode() -> Any:
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.inference_mode()


class EmbeddingSpeakerSimilarityProvider:
    """Compute ASV/SIM from an injected embedding provider."""

    def __init__(self, embedder: EmbeddingProvider, backend: str = "embedding-cosine") -> None:
        self.embedder = embedder
        self.backend = backend

    def __call__(self, prediction: str, reference: str, **kwargs: Any) -> dict[str, Any]:
        score = cosine_similarity(self.embedder(prediction), self.embedder(reference))
        return {
            "ASV": float(score),
            "score": float(score),
            "backend": self.backend,
            "prediction_audio": prediction,
            "reference_audio": reference,
        }


class WavLMSpeakerEmbeddingProvider:
    """WavLM-large embedding provider with lazy Transformers loading."""

    model_id = "microsoft/wavlm-large"

    def __init__(
        self,
        model_id: str | None = None,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id or self.model_id
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.cache_dir = cache_dir
        self._feature_extractor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            configure_model_cache(self.cache_dir)
            if self.cache_dir is not None:
                import os

                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import torch

            # Some base runtime containers ship a broken optional deepspeed install.
            # Transformers can import it while loading modeling_utils; WavLM
            # inference itself does not need it.
            install_deepspeed_stub()
            from transformers import AutoFeatureExtractor, WavLMModel

            load_kwargs: dict[str, Any] = {}
            if self.cache_dir is not None:
                load_kwargs = {
                    "cache_dir": str(Path(self.cache_dir) / "huggingface" / "hub"),
                    "local_files_only": True,
                }
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id, **load_kwargs)
            self._model = WavLMModel.from_pretrained(self.model_id, **load_kwargs)
            if self.checkpoint_path:
                state = torch.load(self.checkpoint_path, map_location="cpu")
                state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
                self._model.load_state_dict(state_dict, strict=False)
            self._model.to(self.device)
            self._model.eval()
        return self._feature_extractor, self._model

    @staticmethod
    def _load_audio(audio_path: str) -> tuple[Any, int]:
        try:
            import librosa

            waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            return waveform, sample_rate
        except ImportError:
            import soundfile as sf

            waveform, sample_rate = sf.read(audio_path)
            if sample_rate != 16000:
                raise RuntimeError("librosa is required to resample non-16kHz audio") from None
            return waveform, sample_rate

    def __call__(self, audio_path: str) -> Embedding:
        import torch

        feature_extractor, model = self._load()
        waveform, sample_rate = self._load_audio(audio_path)
        inputs = feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.detach().cpu().tolist()


class ECAPATDNNEmbeddingProvider:
    """SpeechBrain ECAPA-TDNN embedding provider."""

    model_id = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id or self.model_id
        self.device = device
        self.cache_dir = cache_dir
        self._encoder: Any | None = None

    def _load(self) -> Any:
        if self._encoder is None:
            configure_model_cache(self.cache_dir)
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # pragma: no cover - compatibility for older SpeechBrain.
                from speechbrain.pretrained import EncoderClassifier

            source = self.model_id
            savedir = None
            overrides: dict[str, Any] = {}
            if self.cache_dir is not None:
                cache_path = Path(self.cache_dir)
                local_snapshot_root = (
                    cache_path
                    / "huggingface"
                    / "hub"
                    / f"models--{self.model_id.replace('/', '--')}"
                    / "snapshots"
                )
                local_snapshots = sorted(path for path in local_snapshot_root.glob("*") if path.is_dir())
                if local_snapshots:
                    source = str(local_snapshots[-1])
                    savedir_path = _writable_cache_subdir(cache_path, "speechbrain", self.model_id.replace("/", "--"))
                    _clear_speechbrain_snapshot_link_conflicts(local_snapshots[-1], savedir_path)
                    savedir = str(savedir_path)
                    overrides["pretrained_path"] = source
                else:
                    savedir = str(_writable_cache_subdir(cache_path, "speechbrain", self.model_id.replace("/", "--")))
            self._encoder = EncoderClassifier.from_hparams(
                source=source,
                savedir=savedir,
                overrides=overrides,
                run_opts={"device": self.device},
            )
        return self._encoder

    def __call__(self, audio_path: str) -> Embedding:
        import torch

        encoder = self._load()
        if hasattr(encoder, "encode_file"):
            embedding = encoder.encode_file(audio_path)
        else:
            waveform = encoder.load_audio(audio_path)
            if isinstance(waveform, torch.Tensor) and waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            embedding = encoder.encode_batch(waveform)
        if hasattr(embedding, "squeeze"):
            embedding = embedding.squeeze()
        if isinstance(embedding, torch.Tensor):
            return embedding.detach().cpu().tolist()
        return list(embedding)


def _clear_speechbrain_snapshot_link_conflicts(snapshot_dir: Path, savedir: Path) -> None:
    """Remove stale same-target files that SpeechBrain will recreate as links."""

    for source_path in snapshot_dir.iterdir():
        destination = savedir / source_path.name
        if not destination.exists() and not destination.is_symlink():
            continue
        if destination.is_symlink():
            try:
                if destination.resolve() == source_path.resolve():
                    destination.unlink()
            except FileNotFoundError:
                destination.unlink(missing_ok=True)


class ERes2NetEmbeddingProvider:
    """3D-Speaker ERes2Net embedding provider through ModelScope."""

    model_id = "iic/speech_eres2net_sv_zh-cn_16k-common"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id or self.model_id
        self.device = device
        self.cache_dir = cache_dir
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        if self._pipeline is None:
            configure_model_cache(self.cache_dir)
            install_deepspeed_stub()
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            model = self.model_id
            if self.cache_dir is not None:
                local_model = Path(self.cache_dir) / "modelscope" / "models" / "iic" / "speech_eres2net_sv_zh-cn_16k-common"
                if local_model.exists():
                    model = str(local_model)
            pipeline_device = normalize_modelscope_device(self.device)
            self._pipeline = pipeline(
                task=Tasks.speaker_verification,
                model=model,
                device=pipeline_device,
            )
        return self._pipeline

    @staticmethod
    def _extract_embedding(result: Any) -> Embedding:
        if isinstance(result, dict) and "embs" in result:
            values = result["embs"]
            if hasattr(values, "tolist"):
                values = values.tolist()
            if values and isinstance(values[0], list):
                values = values[0]
            return [float(value) for value in values]
        raise ValueError("ModelScope ERes2Net did not return an embedding payload")

    def __call__(self, audio_path: str) -> Embedding:
        with _torch_inference_mode():
            result = self._load()([audio_path], output_emb=True)
        return self._extract_embedding(result)


class ERes2NetSimilarityProvider:
    """ModelScope ERes2Net pairwise cosine speaker-verification provider."""

    model_id = "iic/speech_eres2net_sv_zh-cn_16k-common"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
        pipeline_factory: Callable[[], Any] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.model_id = model_id or self.model_id
        self.device = device
        self.cache_dir = cache_dir
        self.pipeline_factory = pipeline_factory
        self.embedding_provider = embedding_provider
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        if self._pipeline is None:
            if self.pipeline_factory is not None:
                self._pipeline = self.pipeline_factory()
                return self._pipeline
            configure_model_cache(self.cache_dir)
            install_deepspeed_stub()
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            model = self.model_id
            if self.cache_dir is not None:
                local_model = Path(self.cache_dir) / "modelscope" / "models" / "iic" / "speech_eres2net_sv_zh-cn_16k-common"
                if local_model.exists():
                    model = str(local_model)
            pipeline_device = normalize_modelscope_device(self.device)
            self._pipeline = pipeline(
                task=Tasks.speaker_verification,
                model=model,
                device=pipeline_device,
            )
        return self._pipeline

    @staticmethod
    def _extract_score(result: Any) -> float:
        if isinstance(result, dict):
            for key in ("score", "similarity", "cosine_similarity", "ASV"):
                if key in result:
                    return float(result[key])
        if isinstance(result, (list, tuple)) and result:
            return ERes2NetSimilarityProvider._extract_score(result[0])
        return float(result)

    def __call__(self, prediction: str, reference: str, **kwargs: Any) -> dict[str, Any]:
        try:
            pipe = self._load()
            if _env_flag("SURE_EVAL_ERES2NET_FORCE_EMBEDDING_COSINE"):
                score = self._score_with_loaded_pipeline_embeddings(pipe, prediction, reference)
                backend = "modelscope-eres2net-same-pipeline-embedding-cosine"
                return self._result_row(score, backend, prediction, reference)
            try:
                try:
                    with _torch_inference_mode():
                        result = pipe([reference, prediction])
                    backend = "modelscope-eres2net-cosine"
                except Exception as exc:
                    if self._is_cuda_oom_error(exc):
                        try:
                            score = self._score_with_loaded_pipeline_embeddings(pipe, prediction, reference)
                        except Exception as recovery_exc:
                            raise RuntimeError(
                                f"ERes2Net CUDA OOM recovery failed after OutOfMemoryError: {recovery_exc}"
                            ) from exc
                        backend = "modelscope-eres2net-same-pipeline-embedding-cosine"
                        return self._result_row(score, backend, prediction, reference)
                    if not self._is_sox_dependency_error(exc):
                        raise
                    prepared_reference = self._preprocess_audio_for_modelscope(reference)
                    prepared_prediction = self._preprocess_audio_for_modelscope(prediction)
                    try:
                        with _torch_inference_mode():
                            result = pipe([prepared_reference, prepared_prediction])
                        backend = "modelscope-eres2net-cosine-preprocessed"
                    finally:
                        Path(prepared_reference).unlink(missing_ok=True)
                        Path(prepared_prediction).unlink(missing_ok=True)
            except TypeError:
                with _torch_inference_mode():
                    result = pipe({"audio_in": reference, "audio_ref": prediction})
                backend = "modelscope-eres2net-cosine"
            score = self._extract_score(result)
        except Exception as exc:
            if self._is_cuda_oom_error(exc):
                raise
            if self.embedding_provider is None:
                raise
            score = cosine_similarity(
                self.embedding_provider(prediction),
                self.embedding_provider(reference),
            )
            backend = "modelscope-eres2net-embedding-cosine"
        return {
            "ASV": float(score),
            "score": float(score),
            "backend": backend,
            "prediction_audio": prediction,
            "reference_audio": reference,
        }

    @staticmethod
    def _result_row(score: float, backend: str, prediction: str, reference: str) -> dict[str, Any]:
        return {
            "ASV": float(score),
            "score": float(score),
            "backend": backend,
            "prediction_audio": prediction,
            "reference_audio": reference,
        }

    def _score_with_loaded_pipeline_embeddings(self, pipe: Any, prediction: str, reference: str) -> float:
        prediction_embedding = self._embedding_with_loaded_pipeline(pipe, prediction)
        reference_embedding = self._embedding_with_loaded_pipeline(pipe, reference)
        return cosine_similarity(prediction_embedding, reference_embedding)

    def _embedding_with_loaded_pipeline(self, pipe: Any, audio_path: str) -> Embedding:
        try:
            with _torch_inference_mode():
                return ERes2NetEmbeddingProvider._extract_embedding(pipe([audio_path], output_emb=True))
        except Exception as exc:
            if not self._is_cuda_oom_error(exc):
                raise
            _empty_cuda_cache()
            segment_embeddings = []
            for segment in self._load_audio_segments(audio_path):
                with _torch_inference_mode():
                    segment_embeddings.append(ERes2NetEmbeddingProvider._extract_embedding(pipe([segment], output_emb=True)))
                _empty_cuda_cache()
            return _mean_embedding(segment_embeddings)

    def _load_audio_segments(self, audio_path: str) -> list[Any]:
        import librosa

        max_seconds = _env_float("SURE_EVAL_ERES2NET_SEGMENT_SECONDS", 15.0)
        if max_seconds <= 0:
            raise RuntimeError("SURE_EVAL_ERES2NET_SEGMENT_SECONDS must be positive")
        sample_rate = 16000
        waveform, _sample_rate = librosa.load(audio_path, sr=sample_rate, mono=True)
        samples_per_segment = max(1, int(max_seconds * sample_rate))
        if len(waveform) <= samples_per_segment:
            return [waveform]
        return [
            waveform[start : start + samples_per_segment]
            for start in range(0, len(waveform), samples_per_segment)
            if len(waveform[start : start + samples_per_segment]) > 0
        ]

    @staticmethod
    def _is_sox_dependency_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "libsox" in message or "sox_effects" in message or "torchaudio_sox" in message

    @staticmethod
    def _is_cuda_oom_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "outofmemoryerror" in message or "out of memory" in message or "cuda oom" in message

    @staticmethod
    def _preprocess_audio_for_modelscope(audio_path: str) -> str:
        import librosa
        import soundfile as sf

        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        output = tempfile.NamedTemporaryFile(prefix="sure-eval-eres2net-", suffix=".wav", delete=False)
        output.close()
        sf.write(output.name, waveform, sample_rate)
        return output.name


def _mean_embedding(embeddings: list[Embedding]) -> Embedding:
    if not embeddings:
        raise ValueError("at least one embedding is required")
    width = len(embeddings[0])
    if any(len(embedding) != width for embedding in embeddings):
        raise ValueError("all embeddings must have the same length")
    return [
        sum(float(embedding[index]) for embedding in embeddings) / len(embeddings)
        for index in range(width)
    ]


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {raw_value!r}") from exc


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _empty_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
