"""Trace node for the FunASR Paraformer audio loader contract.

This node intentionally does not materialize a normalized wav. CV3-style
Mandarin evaluation passes the original audio path to ``AutoModel.generate``;
FunASR's internal loader performs decode, channel reduction, and resampling
according to the model frontend sample rate.
"""

from __future__ import annotations

from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "frontend/funasr_loader_16k_mono"
NODE_VERSION = "v1"


def describe_funasr_loader_16k_mono(
    audio_path: str,
    *,
    language: str = "zh",
    role: str = "prediction_audio",
) -> PipelineNodeResult:
    """Return the trace entry for FunASR's internal audio frontend."""

    return PipelineNodeResult(
        stage="frontend",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "audio_path": audio_path,
            "language": language,
            "role": role,
            "loader": "funasr.utils.load_utils.load_audio_text_image_video",
            "caller": "funasr.AutoModel.generate",
            "target_sample_rate": 16000,
            "channel_policy": "mean_to_mono",
            "materialized_audio_path": None,
            "cv3_compatible": True,
        },
        internal_stages=("audio_decode", "channel_mean", "resample_if_needed"),
    )

