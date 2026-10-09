# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Immutable language facts extracted from legacy language classes."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LocaleProfile:
    """Read-only language metadata at runtime; no request state or counters."""

    code: str
    language: str
    language_name_en: str
    language_name_zh: str
    script: str
    spaced_writing: bool
    score_method: str
    is_tts_variant: bool
    is_tts_pipeline: bool
    tts_policy_id: Optional[str]

    @classmethod
    def from_normalizer(cls, code: str, normalizer: Any) -> "LocaleProfile":
        return cls(
            code=code,
            language=str(normalizer.language),
            language_name_en=str(normalizer.language_name_en),
            language_name_zh=str(normalizer.language_name_zh),
            script=str(normalizer.script),
            spaced_writing=bool(normalizer.spaced_writing),
            score_method=str(normalizer.score_method),
            is_tts_variant=code.endswith("_tts"),
            is_tts_pipeline=bool(normalizer.is_tts_pipeline),
            tts_policy_id=getattr(normalizer, "tts_policy_id", None),
        )
