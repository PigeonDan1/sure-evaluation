"""DNSMOS P.808 component scoring node."""

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._audio_quality import MOSProvider, MOSRow, score_mos_backend
from sure_eval.evaluation.nodes.scoring._score_components import ScoreComponentProvider

NODE_ID = "scoring/dnsmos_p808"
NODE_VERSION = "v1"


def score_dnsmos_p808(rows: list[MOSRow], *, provider: MOSProvider) -> PipelineNodeResult:
    return score_mos_backend(
        rows, metric_name="dnsmos_p808", node_id=NODE_ID,
        provider=ScoreComponentProvider(provider, field="P808_MOS"), version=NODE_VERSION,
    )
