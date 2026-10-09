# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Immutable request object after entering the normalization pipeline."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class NormalizationRequest:
    """Snapshot original language id, resolved language, and compatibility kwargs."""

    requested_language: str
    language: str
    mode: str
    options: Mapping[str, object]

    @classmethod
    def create(
        cls,
        requested_language: str,
        language: str,
        options: Mapping[str, object],
    ) -> "NormalizationRequest":
        if not requested_language:
            raise ValueError("language must not be empty")
        copied_options = dict(options)
        copied_options["language"] = language
        mode = copied_options.get("mode") or "asr_eval"
        copied_options["mode"] = mode
        return cls(
            requested_language=requested_language,
            language=language,
            mode=str(mode),
            options=MappingProxyType(copied_options),
        )

    def to_config_kwargs(self) -> dict:
        """Return an isolated copy for legacy normalizer.config()."""

        return dict(self.options)
