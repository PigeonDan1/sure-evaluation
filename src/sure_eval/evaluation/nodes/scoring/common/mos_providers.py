"""Provider implementations for MOS-style TTS metrics."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List

from sure_eval.compat.deepspeed_stub import install_deepspeed_stub
from sure_eval.evaluation.cache import get_cache_dir
from sure_eval.evaluation.nodes.transcription.common.providers import configure_model_cache


CommandRunner = Callable[[List[str]], str]
DNSMOSScorer = Callable[[str], Dict[str, Any]]
DEFAULT_SHARED_TTS_METRIC_CACHE = Path(
    os.environ.get("SURE_TTS_METRIC_SHARED_CACHE", str(get_cache_dir("tts-metrics")))
)


def _default_command_runner(command: list[str]) -> str:
    return subprocess.check_output(command, text=True)


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _temporary_cudnn_disabled(device: str):
    if not device.startswith("cuda"):
        yield
        return
    try:
        import torch
    except Exception:
        yield
        return

    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is None or not hasattr(cudnn, "enabled"):
        yield
        return

    previous = bool(cudnn.enabled)
    cudnn.enabled = False
    try:
        yield
    finally:
        cudnn.enabled = previous


class CommandMOSProvider:
    """Run an external MOS command and normalize JSON or scalar output."""

    def __init__(
        self,
        command: list[str] | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.command = command
        self.command_runner = command_runner or _default_command_runner

    def __call__(
        self,
        prediction: str,
        reference: str,
        *,
        metric: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        command = self._build_command(prediction, metric=metric, **kwargs)
        raw_output = self.command_runner(command).strip()
        parsed = self._parse_output(raw_output, metric)
        parsed["backend"] = "command"
        return parsed

    def _build_command(self, prediction: str, *, metric: str, **kwargs: Any) -> list[str]:
        if self.command is None:
            return [metric, prediction]
        return [part.format(audio_path=prediction, metric=metric, **kwargs) for part in self.command]

    @staticmethod
    def _parse_output(raw_output: str, metric: str) -> dict[str, Any]:
        try:
            loaded = json.loads(raw_output)
        except json.JSONDecodeError:
            loaded = float(raw_output.splitlines()[-1].strip())

        if isinstance(loaded, dict):
            return dict(loaded)
        if metric == "dnsmos":
            return {"OVRL": float(loaded)}
        if metric == "wv-mos":
            return {"mos": float(loaded)}
        if metric == "utmos":
            return {"utmos": float(loaded)}
        return {"score": float(loaded)}


class CV3EvalDNSMOSScorer:
    """CV3-Eval compatible DNSMOS P.835 ONNX scorer."""

    sampling_rate = 16000
    input_length = 9.01

    def __init__(
        self,
        env_root: str | Path,
        *,
        personalized_mos: bool = False,
        providers: list[str] | None = None,
    ) -> None:
        self.env_root = Path(env_root)
        self.personalized_mos = personalized_mos
        self.providers = providers
        self._onnx_sess: Any | None = None
        self._p808_onnx_sess: Any | None = None

    def __call__(self, audio_path: str) -> dict[str, Any]:
        import librosa
        import numpy as np
        import soundfile as sf

        onnx_sess, p808_onnx_sess = self._load_sessions()
        audio, input_fs = sf.read(audio_path)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        fs = self.sampling_rate
        if input_fs != fs:
            audio = librosa.resample(audio, orig_sr=input_fs, target_sr=fs)

        actual_audio_len = len(audio)
        len_samples = int(self.input_length * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - self.input_length) + 1
        hop_len_samples = fs
        sig_raw_values: list[float] = []
        bak_raw_values: list[float] = []
        ovr_raw_values: list[float] = []
        sig_values: list[float] = []
        bak_values: list[float] = []
        ovr_values: list[float] = []
        p808_values: list[float] = []

        for index in range(num_hops):
            audio_seg = audio[
                int(index * hop_len_samples) : int((index + self.input_length) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.asarray(audio_seg, dtype="float32")[np.newaxis, :]
            p808_features = np.asarray(
                self._audio_melspec(audio=audio_seg[:-160]),
                dtype="float32",
            )[np.newaxis, :, :]
            p808_mos = p808_onnx_sess.run(None, {"input_1": p808_features})[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = onnx_sess.run(
                None,
                {"input_1": input_features},
            )[0][0]
            mos_sig, mos_bak, mos_ovr = self._polyfit(
                mos_sig_raw,
                mos_bak_raw,
                mos_ovr_raw,
            )
            sig_raw_values.append(float(mos_sig_raw))
            bak_raw_values.append(float(mos_bak_raw))
            ovr_raw_values.append(float(mos_ovr_raw))
            sig_values.append(float(mos_sig))
            bak_values.append(float(mos_bak))
            ovr_values.append(float(mos_ovr))
            p808_values.append(float(p808_mos))

        if not ovr_values:
            raise ValueError(f"DNSMOS could not score audio: {audio_path}")

        return {
            "filename": audio_path,
            "len_in_sec": float(actual_audio_len / fs),
            "sr": fs,
            "num_hops": num_hops,
            "OVRL_raw": float(np.mean(ovr_raw_values)),
            "SIG_raw": float(np.mean(sig_raw_values)),
            "BAK_raw": float(np.mean(bak_raw_values)),
            "OVRL": float(np.mean(ovr_values)),
            "SIG": float(np.mean(sig_values)),
            "BAK": float(np.mean(bak_values)),
            "P808_MOS": float(np.mean(p808_values)),
        }

    def _load_sessions(self) -> tuple[Any, Any]:
        if self._onnx_sess is None or self._p808_onnx_sess is None:
            import onnxruntime as ort

            session_options: dict[str, Any] = {}
            if self.providers is not None:
                session_options["providers"] = self.providers
            self._onnx_sess = ort.InferenceSession(
                str(self._primary_model_path()),
                **session_options,
            )
            self._p808_onnx_sess = ort.InferenceSession(
                str(self._p808_model_path()),
                **session_options,
            )
        return self._onnx_sess, self._p808_onnx_sess

    def _primary_model_path(self) -> Path:
        subdir = "pDNSMOS" if self.personalized_mos else "DNSMOS"
        return self.env_root / subdir / "sig_bak_ovr.onnx"

    def _p808_model_path(self) -> Path:
        return self.env_root / "DNSMOS" / "model_v8.onnx"

    def _audio_melspec(
        self,
        audio: Any,
        n_mels: int = 120,
        frame_size: int = 320,
        hop_length: int = 160,
        sr: int = 16000,
        to_db: bool = True,
    ) -> Any:
        import librosa
        import numpy as np

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def _polyfit(self, sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
        import numpy as np

        if self.personalized_mos:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))


class DNSMOSProvider:
    """DNSMOS provider backed by a local DNSMOS script or module.

    The reference CV3-Eval flow computes DNSMOS P.835-style scores with ONNX
    models and returns at least an overall score. This wrapper keeps the metric
    definition independent from a specific checkout layout while providing a
    real subprocess-backed execution path for validation images.
    """

    def __init__(
        self,
        command: list[str] | None = None,
        cache_dir: str | Path | None = None,
        command_runner: CommandRunner | None = None,
        scorer: DNSMOSScorer | None = None,
        env_root: str | Path | None = None,
        personalized_mos: bool = False,
    ) -> None:
        self.command = command
        self.cache_dir = cache_dir
        self.command_runner = command_runner or _default_command_runner
        self._has_custom_command_runner = command_runner is not None
        self.scorer = scorer
        self.env_root = Path(env_root) if env_root is not None else None
        self.personalized_mos = personalized_mos

    def __call__(self, prediction: str, reference: str = "", **kwargs: Any) -> dict[str, Any]:
        configure_model_cache(self.cache_dir)
        if self.scorer is not None:
            row = dict(self.scorer(prediction))
            row["backend"] = row.get("backend", "cv3-eval-dnsmos")
            return row
        env_root = None if self.command is not None or self._has_custom_command_runner else self._resolve_env_root()
        if env_root is not None:
            self.scorer = CV3EvalDNSMOSScorer(
                env_root,
                personalized_mos=self.personalized_mos,
            )
            row = dict(self.scorer(prediction))
            row["backend"] = row.get("backend", "cv3-eval-dnsmos")
            return row

        command = self.command or ["dnsmos", "{audio_path}"]
        row = CommandMOSProvider(
            command=command,
            command_runner=self.command_runner,
        )(prediction, reference, metric="dnsmos", **kwargs)
        row["backend"] = row.get("backend", "dnsmos-command")
        return row

    def _resolve_env_root(self) -> Path | None:
        candidates: list[Path] = []
        if self.env_root is not None:
            candidates.append(self.env_root)
        if self.cache_dir is not None:
            candidates.extend(
                [
                    Path(self.cache_dir) / "dnsmos",
                    Path(self.cache_dir) / "DNSMOS",
                    Path(self.cache_dir),
                ]
            )
        candidates.extend(
            [
                DEFAULT_SHARED_TTS_METRIC_CACHE / "mos" / "dnsmos",
                DEFAULT_SHARED_TTS_METRIC_CACHE / "mos" / "DNSMOS",
                DEFAULT_SHARED_TTS_METRIC_CACHE / "mos",
            ]
        )
        for candidate in candidates:
            if (
                (candidate / "DNSMOS" / "model_v8.onnx").exists()
                and (candidate / "DNSMOS" / "sig_bak_ovr.onnx").exists()
            ):
                return candidate
        return None


class WVMOSProvider:
    """WV-MOS provider backed by EmergentTTS-Eval/Wav2Vec2MOS code."""

    def __init__(
        self,
        command: list[str] | None = None,
        cache_dir: str | Path | None = None,
        command_runner: CommandRunner | None = None,
        repo_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.command = command
        self.cache_dir = cache_dir
        self.command_runner = command_runner or _default_command_runner
        self.repo_dir = Path(repo_dir) if repo_dir is not None else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.device = device
        self._model: Any | None = None

    def __call__(self, prediction: str, reference: str = "", **kwargs: Any) -> dict[str, Any]:
        configure_model_cache(self.cache_dir)
        if self.command is not None:
            row = CommandMOSProvider(
                command=self.command,
                command_runner=self.command_runner,
            )(prediction, reference, metric="wv-mos", **kwargs)
            row["backend"] = row.get("backend", "wv-mos-command")
            return row

        model = self._load_model()
        with _temporary_cudnn_disabled(self.device):
            score = float(model.calculate_one(prediction, self.device))
        return {"mos": score, "score": score, "backend": "emergenttts-wv-mos"}

    def _load_model(self) -> Any:
        if self._model is None:
            repo_dir = self._resolve_repo_dir()
            checkpoint_path = self._resolve_checkpoint_path()
            configure_model_cache(self.cache_dir)
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            # Some base runtime containers contain a broken optional deepspeed install.
            # Transformers can import it while loading Wav2Vec2; WV-MOS does not
            # need deepspeed for inference.
            install_deepspeed_stub()
            sys.path.insert(0, str(repo_dir))
            try:
                from quality_assessment.wv_mos import Wav2Vec2MOS
            finally:
                try:
                    sys.path.remove(str(repo_dir))
                except ValueError:
                    pass
            self._model = Wav2Vec2MOS(str(checkpoint_path))
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def _resolve_repo_dir(self) -> Path:
        if self.repo_dir is not None:
            return self.repo_dir
        for root in self._candidate_cache_dirs():
            candidate = root / "repos" / "emergenttts-eval-public"
            if candidate.exists():
                return candidate
        raise FileNotFoundError("EmergentTTS-Eval repo_dir is required for WV-MOS inference")

    def _resolve_checkpoint_path(self) -> Path:
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        for root in self._candidate_cache_dirs():
            candidate = root / "wv-mos" / "wav2vec2.ckpt"
            if candidate.exists():
                return candidate
        raise FileNotFoundError("WV-MOS wav2vec2.ckpt checkpoint is required")

    def _candidate_cache_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        if self.cache_dir is not None:
            candidates.append(Path(self.cache_dir))
        candidates.append(DEFAULT_SHARED_TTS_METRIC_CACHE / "mos")
        return candidates


class UTMOSProvider:
    """UTMOS22 provider backed by a local UTMOS command or Space checkout."""

    def __init__(
        self,
        command: list[str] | None = None,
        cache_dir: str | Path | None = None,
        command_runner: CommandRunner | None = None,
        repo_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.command = command
        self.cache_dir = cache_dir
        self.command_runner = command_runner or _default_command_runner
        self.repo_dir = Path(repo_dir) if repo_dir is not None else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.device = device
        self._scorer: Any | None = None

    def __call__(self, prediction: str, reference: str = "", **kwargs: Any) -> dict[str, Any]:
        configure_model_cache(self.cache_dir)
        if self.command is not None:
            row = CommandMOSProvider(
                command=self.command,
                command_runner=self.command_runner,
            )(prediction, reference, metric="utmos", **kwargs)
            row["backend"] = row.get("backend", "utmos-command")
            return row

        scorer = self._load_scorer(prediction)
        import torchaudio

        waveform, _sample_rate = torchaudio.load(prediction)
        score_values = scorer.score(waveform.to(self.device))
        score = float(score_values[0])
        return {"utmos": score, "score": score, "backend": "utmos22-demo"}

    def _load_scorer(self, prediction: str) -> Any:
        if self._scorer is None:
            repo_dir = self._resolve_repo_dir()
            checkpoint_path = self._resolve_checkpoint_path()
            sys.path.insert(0, str(repo_dir))
            try:
                from score import Score
                import torchaudio
            finally:
                try:
                    sys.path.remove(str(repo_dir))
                except ValueError:
                    pass
            _waveform, sample_rate = torchaudio.load(prediction)
            with _pushd(repo_dir):
                self._scorer = Score(
                    ckpt_path=str(checkpoint_path),
                    input_sample_rate=int(sample_rate),
                    device=self.device,
                )
        return self._scorer

    def _resolve_repo_dir(self) -> Path:
        if self.repo_dir is not None:
            return self.repo_dir
        for root in self._candidate_cache_dirs():
            for candidate in (
                root / "utmos22" / "UTMOS-demo",
                root / "repos" / "UTMOS-demo",
            ):
                if candidate.exists():
                    return candidate
        raise FileNotFoundError("UTMOS-demo repo_dir is required for UTMOS inference")

    def _resolve_checkpoint_path(self) -> Path:
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        repo_dir = self._resolve_repo_dir()
        candidate = repo_dir / "epoch=3-step=7459.ckpt"
        if candidate.exists():
            return candidate
        raise FileNotFoundError("UTMOS epoch=3-step=7459.ckpt checkpoint is required")

    def _candidate_cache_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        if self.cache_dir is not None:
            candidates.append(Path(self.cache_dir))
        candidates.append(DEFAULT_SHARED_TTS_METRIC_CACHE / "mos")
        return candidates


def build_python_command(module_or_script: str, *args: str) -> list[str]:
    """Build a Python command for metric scripts while preserving templates."""
    python = os.environ.get("PYTHON", "python")
    if module_or_script.endswith(".py"):
        return [python, module_or_script, *args]
    return [python, "-m", module_or_script, *args]
