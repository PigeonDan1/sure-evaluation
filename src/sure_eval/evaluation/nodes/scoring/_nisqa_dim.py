"""NISQA v2 DIM predictions from a manually staged checkpoint."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any


_MODEL_ARGS = (
    "ms_seg_length", "ms_n_mels", "cnn_model", "cnn_c_out_1", "cnn_c_out_2",
    "cnn_c_out_3", "cnn_kernel_size", "cnn_dropout", "cnn_pool_1", "cnn_pool_2",
    "cnn_pool_3", "cnn_fc_out_h", "td", "td_sa_d_model", "td_sa_nhead",
    "td_sa_pos_enc", "td_sa_num_layers", "td_sa_h", "td_sa_dropout",
    "td_lstm_h", "td_lstm_num_layers", "td_lstm_dropout", "td_lstm_bidirectional",
    "td_2", "td_2_sa_d_model", "td_2_sa_nhead", "td_2_sa_pos_enc",
    "td_2_sa_num_layers", "td_2_sa_h", "td_2_sa_dropout", "td_2_lstm_h",
    "td_2_lstm_num_layers", "td_2_lstm_dropout", "td_2_lstm_bidirectional",
    "pool", "pool_att_h", "pool_att_dropout",
)
_FIELDS = ("nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud")


class NISQADIMProvider:
    """NISQA DIM predicts five scores from enhanced speech alone."""

    def __init__(self, *, checkpoint: str | None = None, device: str = "cpu") -> None:
        self.checkpoint = checkpoint or os.environ.get("SURE_SE_NISQA_DIM_CKPT")
        self.device = device
        self._model: Any | None = None
        self._args: dict[str, Any] | None = None
        self._scores: dict[str, dict[str, float | str]] = {}

    def _load_model(self) -> tuple[Any, dict[str, Any]]:
        if self._model is None:
            if not self.checkpoint or not Path(self.checkpoint).is_file():
                raise RuntimeError("NISQA DIM requires SURE_SE_NISQA_DIM_CKPT pointing to nisqa.tar")
            import torch
            from nisqa import NISQA_lib as lib

            checkpoint = torch.load(self.checkpoint, map_location="cpu")
            args = checkpoint["args"]
            if args.get("model") != "NISQA_DIM":
                raise ValueError("NISQA checkpoint must be the five-dimensional NISQA_DIM model")
            model = lib.NISQA_DIM(**{name: args[name] for name in _MODEL_ARGS})
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self._model = model.to(self.device).eval()
            self._args = args
        assert self._args is not None
        return self._model, self._args

    def score_batch(self, rows: list[tuple[str, str]], *, metric_name: str) -> list[dict[str, Any]]:
        import pandas as pd
        from nisqa import NISQA_lib as lib

        if not rows:
            return []
        pending = [audio for _sample_id, audio in rows if audio not in self._scores]
        if pending:
            model, args = self._load_model()
            frame = pd.DataFrame({"deg": [str(Path(path).resolve()) for path in pending]})
            dataset = lib.SpeechQualityDataset(
                frame, df_con=None, data_dir="", filename_column="deg",
                mos_column="predict_only", seg_length=args["ms_seg_length"],
                max_length=args["ms_max_segments"], to_memory=None,
                to_memory_workers=None, seg_hop_length=args["ms_seg_hop_length"],
                transform=None, ms_n_fft=args["ms_n_fft"],
                ms_hop_length=args["ms_hop_length"], ms_win_length=args["ms_win_length"],
                ms_n_mels=args["ms_n_mels"], ms_sr=args["ms_sr"],
                ms_fmax=args["ms_fmax"], ms_channel=None, double_ended=False,
                dim=True, filename_column_ref=None,
            )
            lib.predict_dim(model, dataset, bs=8, dev=self.device, num_workers=0)
            for index, path in enumerate(pending):
                values = {
                    field: float(dataset.df[f"{field.removeprefix('nisqa_')}_pred"].iloc[index])
                    for field in _FIELDS
                }
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError(f"NISQA returned a non-finite prediction for {path}")
                self._scores[path] = {**values, "backend": "nisqa-v2-dim"}
        return [dict(self._scores[audio]) for _sample_id, audio in rows]

    def __call__(self, prediction: str, reference: str = "", **kwargs: Any) -> dict[str, Any]:
        return self.score_batch([("", prediction)], metric_name="nisqa_mos")[0]
