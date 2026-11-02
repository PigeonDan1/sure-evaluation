"""SQUIM objective PESQ prediction scoring node."""

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._audio_quality import MOSProvider, MOSRow, score_mos_backend
from sure_eval.evaluation.nodes.scoring._score_components import ScoreComponentProvider

NODE_ID = "scoring/squim_pesq"
NODE_VERSION = "v1"


def score_squim_pesq(rows: list[MOSRow], *, provider: MOSProvider) -> PipelineNodeResult:
    return score_mos_backend(
        rows, metric_name="squim_pesq", node_id=NODE_ID,
        provider=ScoreComponentProvider(provider, field="squim_pesq"), version=NODE_VERSION,
    )
