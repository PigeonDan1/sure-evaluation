"""Cosine scoring for speaker-verification trial manifests."""

from sure_eval.evaluation.nodes.scoring.cosine_trial_scores.node import (
    SVScoreArtifacts,
    load_trial_manifest,
    score_cosine_trials,
)

__all__ = ["SVScoreArtifacts", "load_trial_manifest", "score_cosine_trials"]
