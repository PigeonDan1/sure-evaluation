"""Extended STOI scoring node."""

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._full_reference_audio import (
    ESTOIProvider,
    FullReferenceAudioProvider,
    FullReferenceAudioRow,
    score_full_reference_audio_backend,
)

NODE_ID = "scoring/estoi"
NODE_VERSION = "v1"


def score_estoi(
    rows: list[FullReferenceAudioRow], *, provider: FullReferenceAudioProvider | None = None
) -> PipelineNodeResult:
    return score_full_reference_audio_backend(
        rows, metric_name="estoi", node_id=NODE_ID,
        provider=provider or ESTOIProvider(), version=NODE_VERSION,
    )
