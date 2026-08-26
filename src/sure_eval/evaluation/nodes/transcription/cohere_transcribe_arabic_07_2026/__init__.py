"""Cohere Transcribe Arabic transcription node."""

__all__ = ["transcribe_cohere_transcribe_arabic_07_2026"]


def __getattr__(name: str):
    if name in __all__:
        from sure_eval.evaluation.nodes.transcription.cohere_transcribe_arabic_07_2026 import node

        return getattr(node, name)
    raise AttributeError(name)
