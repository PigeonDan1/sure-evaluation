"""NISQA DIM loudness scoring node."""

from sure_eval.evaluation.nodes.scoring._audio_quality import MOSProvider, MOSRow, score_mos_backend
from sure_eval.evaluation.nodes.scoring._score_components import ScoreComponentProvider

NODE_ID = "scoring/nisqa_loud"
NODE_VERSION = "v1"


def score_nisqa_loud(rows: list[MOSRow], *, provider: MOSProvider):
    return score_mos_backend(rows, metric_name="nisqa_loud", node_id=NODE_ID,
                             provider=ScoreComponentProvider(provider, field="nisqa_loud"), version=NODE_VERSION)
