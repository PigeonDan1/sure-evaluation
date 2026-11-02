"""NISQA DIM noise scoring node."""

from sure_eval.evaluation.nodes.scoring._audio_quality import MOSProvider, MOSRow, score_mos_backend
from sure_eval.evaluation.nodes.scoring._score_components import ScoreComponentProvider

NODE_ID = "scoring/nisqa_noi"
NODE_VERSION = "v1"


def score_nisqa_noi(rows: list[MOSRow], *, provider: MOSProvider):
    return score_mos_backend(rows, metric_name="nisqa_noi", node_id=NODE_ID,
                             provider=ScoreComponentProvider(provider, field="nisqa_noi"), version=NODE_VERSION)
