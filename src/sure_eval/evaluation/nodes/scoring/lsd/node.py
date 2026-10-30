"""Log-spectral distance scoring node."""

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._full_reference_audio import (
    FullReferenceAudioProvider,
    FullReferenceAudioRow,
    LSDProvider,
    score_full_reference_audio_backend,
)

NODE_ID = "scoring/lsd"
NODE_VERSION = "v1"


def score_lsd(
    rows: list[FullReferenceAudioRow], *, provider: FullReferenceAudioProvider | None = None
) -> PipelineNodeResult:
    return score_full_reference_audio_backend(
        rows, metric_name="lsd", node_id=NODE_ID,
        provider=provider or LSDProvider(), version=NODE_VERSION,
    )
