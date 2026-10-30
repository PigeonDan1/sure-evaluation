"""URGENT 2025 phoneme-similarity metric for paired speech audio."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._full_reference_audio import (
    FullReferenceAudioProvider, FullReferenceAudioRow, score_full_reference_audio_backend,
)

NODE_ID = "scoring/lps"
NODE_VERSION = "v1"


def levenshtein(left: str, right: str) -> int:
    """Unit-cost Unicode-character Levenshtein distance used by URGENT LPS."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        current = [index]
        for column, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1, previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


class LPSProvider:
    """Pinned wav2vec2-espeak phoneme recognizer for URGENT LPS."""

    def __init__(self, *, model_dir: str | None = None, device: str = "cpu") -> None:
        self.model_dir = model_dir or os.environ.get("SURE_SE_LPS_MODEL_DIR")
        self.device = device
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None:
            if not self.model_dir or not Path(self.model_dir, "pytorch_model.bin").is_file():
                raise RuntimeError("LPS requires SURE_SE_LPS_MODEL_DIR with the official wav2vec2-espeak checkpoint")
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            self._processor = Wav2Vec2Processor.from_pretrained(
                self.model_dir, local_files_only=True, do_phonemize=False,
            )
            self._model = Wav2Vec2ForCTC.from_pretrained(
                self.model_dir, local_files_only=True,
            ).to(self.device).eval()
        return self._processor, self._model

    @staticmethod
    def _wave(path: str) -> np.ndarray:
        import soundfile as sf
        import soxr

        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1)
        if rate != 16000:
            mono = soxr.resample(mono, rate, 16000)
        if not len(mono):
            raise ValueError("LPS requires non-empty audio")
        return np.asarray(mono, dtype=np.float32)

    def _phonemes(self, audio: np.ndarray) -> str:
        import torch

        processor, model = self._load_model()
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
        with torch.inference_mode():
            logits = model(inputs.input_values.to(self.device)).logits
        ids = torch.argmax(logits, dim=-1)
        return str(processor.batch_decode(ids)[0]).replace(" ", "")

    def __call__(self, enhanced_audio: str, reference_audio: str, **kwargs: Any) -> dict[str, Any]:
        predicted = self._phonemes(self._wave(enhanced_audio))
        reference = self._phonemes(self._wave(reference_audio))
        if not reference:
            raise ValueError("LPS is undefined for an empty reference phoneme string")
        score = 1.0 - levenshtein(predicted, reference) / len(reference)
        if not math.isfinite(score):
            raise ValueError("LPS score is non-finite")
        return {"lps": score, "reference_phonemes": reference, "predicted_phonemes": predicted}


def score_lps(
    rows: list[FullReferenceAudioRow], *, provider: FullReferenceAudioProvider | None = None
) -> PipelineNodeResult:
    return score_full_reference_audio_backend(
        rows, metric_name="lps", node_id=NODE_ID,
        provider=provider or LPSProvider(), version=NODE_VERSION,
    )
