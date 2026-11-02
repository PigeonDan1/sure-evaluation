"""Expose versioned scores from a shared multi-output audio provider."""

from __future__ import annotations

import math
from typing import Any

from sure_eval.evaluation.nodes.scoring._audio_quality import MOSProvider


class ScoreComponentProvider:
    """Select one component without changing upstream inference or calibration."""

    def __init__(self, provider: MOSProvider, *, field: str) -> None:
        self.provider = provider
        self.field = field

    def _select(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.field not in row:
            raise KeyError(f"score provider did not return required field {self.field!r}")
        score = float(row[self.field])
        if not math.isfinite(score):
            raise ValueError(f"{self.field} score is non-finite")
        return {"score": score, self.field: score, "backend": row.get("backend", "multi-output")}

    def __call__(self, prediction: str, reference: str = "", **kwargs: Any) -> dict[str, Any]:
        return self._select(self.provider(prediction, reference, **kwargs))

    def score_batch(self, rows: list[tuple[str, str]], *, metric_name: str) -> list[dict[str, Any]]:
        batch = getattr(self.provider, "score_batch", None)
        if callable(batch):
            result = batch(rows, metric_name=metric_name)
        else:
            result = [self.provider(prediction, "") for _sample_id, prediction in rows]
        if len(result) != len(rows):
            raise RuntimeError(f"score provider returned {len(result)} rows for {len(rows)} inputs")
        return [self._select(dict(item)) for item in result]
