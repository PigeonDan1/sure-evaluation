"""Provider implementations for speaker-similarity metrics."""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import nullcontext
from typing import Callable, Sequence
from pathlib import Path
from typing import Any

from sure_eval.compat.deepspeed_stub import install_deepspeed_stub
from sure_eval.evaluation.nodes.transcription.common.providers import (
    configure_model_cache,
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

    dot = sum(left * right for left, right in zip(left_values, right_values, strict=True))
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


def convert_seed_wavlm_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert Seed-TTS-Eval WavLM SV checkpoint keys to local module keys."""

    converted: dict[str, Any] = {}
    skip_prefixes = ("loss_calculator.",)
    for key, value in state_dict.items():
        if key.startswith(skip_prefixes):
            continue
        if key == "feature_weight":
            converted[key] = value
            continue
        if not key.startswith("feature_extract.model."):
            converted[key] = value
            continue

        inner = key.removeprefix("feature_extract.model.")
        mapped = _convert_seed_wavlm_backbone_key(inner)
        if mapped is not None:
            converted[f"wavlm.{mapped}"] = value
    return converted


def _convert_seed_wavlm_backbone_key(key: str) -> str | None:
    if key == "mask_emb":
        return "masked_spec_embed"
    if key.startswith("layer_norm."):
        return f"feature_projection.layer_norm.{key.removeprefix('layer_norm.')}"
    if key.startswith("post_extract_proj."):
        return f"feature_projection.projection.{key.removeprefix('post_extract_proj.')}"
    if key.startswith("encoder.pos_conv.0."):
        suffix = key.removeprefix("encoder.pos_conv.0.")
        if suffix == "weight_g":
            return "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        if suffix == "weight_v":
            return "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        return f"encoder.pos_conv_embed.conv.{suffix}"
    if key.startswith("encoder.layer_norm."):
        return key

    conv_match = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.(.+)", key)
    if conv_match:
        layer_index, suffix = conv_match.groups()
        if suffix == "0.weight":
            return f"feature_extractor.conv_layers.{layer_index}.conv.weight"
        norm_match = re.fullmatch(r"2\.1\.(weight|bias)", suffix)
        if norm_match:
            return f"feature_extractor.conv_layers.{layer_index}.layer_norm.{norm_match.group(1)}"
        return None

    layer_match = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", key)
    if not layer_match:
        return None

    layer_index, suffix = layer_match.groups()
    prefix = f"encoder.layers.{layer_index}"
    if suffix.startswith("self_attn."):
        attn_suffix = suffix.removeprefix("self_attn.")
        if attn_suffix == "grep_a":
            return f"{prefix}.attention.gru_rel_pos_const"
        if attn_suffix.startswith("grep_linear."):
            return f"{prefix}.attention.gru_rel_pos_linear.{attn_suffix.removeprefix('grep_linear.')}"
        if attn_suffix.startswith("relative_attention_bias."):
            return f"{prefix}.attention.rel_attn_embed.{attn_suffix.removeprefix('relative_attention_bias.')}"
        return f"{prefix}.attention.{attn_suffix}"
    if suffix.startswith("self_attn_layer_norm."):
        return f"{prefix}.layer_norm.{suffix.removeprefix('self_attn_layer_norm.')}"
    if suffix.startswith("fc1."):
        return f"{prefix}.feed_forward.intermediate_dense.{suffix.removeprefix('fc1.')}"
    if suffix.startswith("fc2."):
        return f"{prefix}.feed_forward.output_dense.{suffix.removeprefix('fc2.')}"
    if suffix.startswith("final_layer_norm."):
        return f"{prefix}.final_layer_norm.{suffix.removeprefix('final_layer_norm.')}"
    return None


class SeedWavLMSpeakerEmbeddingProvider:
    """Seed-TTS-Eval compatible WavLM-large speaker embedding provider."""

    model_id = "microsoft/wavlm-large"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        model_id: str | None = None,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id or self.model_id
        self.checkpoint_path = Path(checkpoint_path)
        self.device = normalize_modelscope_device(device) or "cpu"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._model: Any | None = None
        self.load_report: dict[str, list[str]] = {"missing_keys": [], "unexpected_keys": []}

    def _load(self) -> Any:
        if self._model is None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"WavLM Seed checkpoint not found: {self.checkpoint_path}")
            configure_model_cache(self.cache_dir)
            if self.cache_dir is not None:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            import torch
            from transformers import WavLMConfig

            config_kwargs: dict[str, Any] = {}
            if self.cache_dir is not None:
                config_kwargs = {
                    "cache_dir": str(self.cache_dir / "huggingface" / "hub"),
                    "local_files_only": True,
                }
            config = WavLMConfig.from_pretrained(self.model_id, **config_kwargs)
            config.output_hidden_states = True
            model = _create_seed_wavlm_ecapa_model(config)

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            raw_state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            load_result = model.load_state_dict(convert_seed_wavlm_state_dict(raw_state), strict=False)
            self.load_report = {
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
            }
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    @staticmethod
    def _load_audio(audio_path: str) -> Any:
        import librosa

        waveform, _sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        return waveform

    def __call__(self, audio_path: str) -> Embedding:
        import torch

        model = self._load()
        waveform = self._load_audio(audio_path)
        inputs = torch.as_tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            embedding = model(inputs).squeeze(0)
        return embedding.detach().cpu().tolist()


def _build_seed_model_classes() -> tuple[type[Any], type[Any], type[Any], type[Any], type[Any], type[Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import WavLMModel

    class Conv1dReluBn(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            bias: bool = True,
        ) -> None:
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
            self.bn = nn.BatchNorm1d(out_channels)

        def forward(self, x: Any) -> Any:
            return self.bn(F.relu(self.conv(x)))

    class Res2Conv1dReluBn(nn.Module):
        def __init__(
            self,
            channels: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            bias: bool = True,
            scale: int = 4,
        ) -> None:
            super().__init__()
            if channels % scale != 0:
                raise ValueError(f"{channels} must be divisible by scale {scale}")
            self.scale = scale
            self.width = channels // scale
            self.nums = scale if scale == 1 else scale - 1
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
                    for _ in range(self.nums)
                ]
            )
            self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

        def forward(self, x: Any) -> Any:
            chunks = torch.split(x, self.width, dim=1)
            outputs = []
            running = None
            for index in range(self.nums):
                running = chunks[index] if index == 0 else running + chunks[index]
                running = self.bns[index](F.relu(self.convs[index](running)))
                outputs.append(running)
            if self.scale != 1:
                outputs.append(chunks[self.nums])
            return torch.cat(outputs, dim=1)

    class SEConnect(nn.Module):
        def __init__(self, channels: int, se_bottleneck_dim: int = 128) -> None:
            super().__init__()
            self.linear1 = nn.Linear(channels, se_bottleneck_dim)
            self.linear2 = nn.Linear(se_bottleneck_dim, channels)

        def forward(self, x: Any) -> Any:
            weights = x.mean(dim=2)
            weights = F.relu(self.linear1(weights))
            weights = torch.sigmoid(self.linear2(weights))
            return x * weights.unsqueeze(2)

    class SERes2Block(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int,
            stride: int,
            padding: int,
            dilation: int,
            scale: int,
            se_bottleneck_dim: int,
        ) -> None:
            super().__init__()
            self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1)
            self.Res2Conv1dReluBn = Res2Conv1dReluBn(
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                scale=scale,
            )
            self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1)
            self.SE_Connect = SEConnect(out_channels, se_bottleneck_dim)
            self.shortcut = None
            if in_channels != out_channels:
                self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        def forward(self, x: Any) -> Any:
            residual = self.shortcut(x) if self.shortcut is not None else x
            x = self.Conv1dReluBn1(x)
            x = self.Res2Conv1dReluBn(x)
            x = self.Conv1dReluBn2(x)
            x = self.SE_Connect(x)
            return x + residual

    class AttentiveStatsPool(nn.Module):
        def __init__(self, in_dim: int, attention_channels: int = 128, global_context_att: bool = False) -> None:
            super().__init__()
            self.global_context_att = global_context_att
            input_dim = in_dim * 3 if global_context_att else in_dim
            self.linear1 = nn.Conv1d(input_dim, attention_channels, kernel_size=1)
            self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

        def forward(self, x: Any) -> Any:
            if self.global_context_att:
                mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
                std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
                x_in = torch.cat((x, mean, std), dim=1)
            else:
                x_in = x
            alpha = torch.tanh(self.linear1(x_in))
            alpha = torch.softmax(self.linear2(alpha), dim=2)
            mean = torch.sum(alpha * x, dim=2)
            residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
            std = torch.sqrt(residuals.clamp(min=1e-9))
            return torch.cat([mean, std], dim=1)

    class SeedWavLMECAPAModel(nn.Module):
        def __init__(self, config: Any) -> None:
            super().__init__()
            self.wavlm = WavLMModel(config)
            feat_dim = int(config.hidden_size)
            channels = 512
            final_channels = 1536
            self.feature_weight = nn.Parameter(torch.zeros(int(config.num_hidden_layers) + 1))
            self.instance_norm = nn.InstanceNorm1d(feat_dim)
            self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
            self.layer2 = SERes2Block(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                scale=8,
                se_bottleneck_dim=128,
            )
            self.layer3 = SERes2Block(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
                scale=8,
                se_bottleneck_dim=128,
            )
            self.layer4 = SERes2Block(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
                scale=8,
                se_bottleneck_dim=128,
            )
            self.conv = nn.Conv1d(channels * 3, final_channels, kernel_size=1)
            self.pooling = AttentiveStatsPool(final_channels, attention_channels=128)
            self.bn = nn.BatchNorm1d(final_channels * 2)
            self.linear = nn.Linear(final_channels * 2, 256)

        def forward(self, waveform: Any) -> Any:
            outputs = self.wavlm(waveform, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("WavLM did not return hidden states")
            weights = F.softmax(self.feature_weight[: len(hidden_states)], dim=-1)
            stacked = torch.stack(hidden_states, dim=0)
            x = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6
            x = self.instance_norm(x)
            out1 = self.layer1(x)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out = torch.cat([out2, out3, out4], dim=1)
            out = F.relu(self.conv(out))
            out = self.bn(self.pooling(out))
            return self.linear(out)

    return Conv1dReluBn, Res2Conv1dReluBn, SEConnect, SERes2Block, AttentiveStatsPool, SeedWavLMECAPAModel


def _create_seed_wavlm_ecapa_model(config: Any) -> Any:
    return _build_seed_model_classes()[-1](config)


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
