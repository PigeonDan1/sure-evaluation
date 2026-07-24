"""Mixed / English token error rate scoring for canonical ASR routes.

Applies the exact same token-level scorer as ``scoring/token_cer``: one
scorer and one tokenizer. Route differences live in the normalization stage
and public pipeline IDs. For en/cs routes,
``normalization/canonical_itn`` additionally whisper-normalizes latin spans
(contraction expansion, spoken numbers -> digits, spoken-filler removal,
British->American spelling) before the shared canonical chain.

Because the scorer and tokenizer are shared, the degeneration guarantees
hold by construction: text without latin letters scores identically under
the canonical ASR MER and CER pipeline IDs, and text without CJK scores
identically under the canonical ASR MER and WER pipeline IDs.
"""

from __future__ import annotations

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.scoring.token_cer.node import score_key_text_tokens

NODE_ID = "scoring/token_mer"
NODE_VERSION = "v1"
INTERNAL_STAGES = ("canonical_tokenization", "token_edit_distance", "sdi_decomposition")


def score_token_mer(
    files: KeyTextFiles,
    *,
    metric: str = "mer_canonical",
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score canonical key-text files with mixed-token edit distance."""

    return score_key_text_tokens(
        files,
        metric=metric,
        score_key="mer" if metric == "mer_canonical" else "wer",
        node_id=NODE_ID,
        version=NODE_VERSION,
        internal_stages=INTERNAL_STAGES,
    )
