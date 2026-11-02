"""ETC-Bench-compatible evaluation-side transcript canonicalization."""

from .node import (
    NODE_ID,
    NODE_VERSION,
    SUPPORTED_PROFILES,
    normalize_etc_key_text_files,
    normalize_etc_text,
)

__all__ = [
    "NODE_ID",
    "NODE_VERSION",
    "SUPPORTED_PROFILES",
    "normalize_etc_key_text_files",
    "normalize_etc_text",
]
