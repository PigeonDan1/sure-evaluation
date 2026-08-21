"""NeMo Arabic text normalization node."""

__all__ = ["normalize_nemo_key_text_files", "normalize_nemo_text"]


def __getattr__(name: str):
    if name in __all__:
        from sure_eval.evaluation.nodes.normalization.nemo_norm import node

        return getattr(node, name)
    raise AttributeError(name)
