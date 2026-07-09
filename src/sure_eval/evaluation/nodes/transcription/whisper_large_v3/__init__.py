"""Whisper-large-v3 transcription node."""

from sure_eval.evaluation.nodes.transcription.whisper_large_v3.node import (
    NODE_ID,
    NODE_VERSION,
    transcribe_whisper_large_v3,
)

__all__ = ["NODE_ID", "NODE_VERSION", "transcribe_whisper_large_v3"]
