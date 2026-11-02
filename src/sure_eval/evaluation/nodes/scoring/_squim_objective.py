"""TorchAudio SQUIM objective predictions from an explicitly staged checkpoint."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any


class SQUIMObjectiveProvider:
    """Return the three SQUIM objective outputs, cached per audio path."""

    def __init__(self, *, checkpoint: str | None = None, device: str = "cpu") -> None:
        self.checkpoint = checkpoint or os.environ.get("SURE_SE_SQUIM_OBJECTIVE_CKPT")
        self.device = device
        self._model: Any | None = None
        self._scores: dict[str, dict[str, float | str]] = {}

    def _load_model(self) -> Any:
        if self._model is None:
            if not self.checkpoint or not Path(self.checkpoint).is_file():
                raise RuntimeError("SQUIM requires SURE_SE_SQUIM_OBJECTIVE_CKPT pointing to squim_objective_dns2020.pth")
            import torch
            from torchaudio.models import squim_objective_base

            model = squim_objective_base()
            model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"), strict=True)
            self._model = model.to(self.device).eval()
        return self._model

    def __call__(self, prediction: str, reference: str = "", **kwargs: Any) -> dict[str, Any]:
        if prediction not in self._scores:
            import torch
            import torchaudio

            waveform, rate = torchaudio.load(prediction)
            waveform = waveform.mean(dim=0, keepdim=True)
            if rate != 16000:
                waveform = torchaudio.functional.resample(waveform, rate, 16000)
            if waveform.shape[1] == 0:
                raise ValueError("SQUIM requires non-empty audio")
            with torch.inference_mode():
                scores = self._load_model()(waveform.to(self.device))
            values = [float(item.detach().cpu().reshape(-1)[0]) for item in scores]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("SQUIM returned a non-finite prediction")
            self._scores[prediction] = {
                "squim_stoi": values[0], "squim_pesq": values[1],
                "squim_si_sdr": values[2], "backend": "torchaudio-squim-objective-dns2020",
            }
        return dict(self._scores[prediction])

    def score_batch(self, rows: list[tuple[str, str]], *, metric_name: str) -> list[dict[str, Any]]:
        return [self(prediction) for _sample_id, prediction in rows]
