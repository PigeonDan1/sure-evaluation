# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Adapter for long-tail TN rule engines."""

from .nemo import (
    DEFAULT_NEMO_CACHE_DIR,
    NEMO_BACKENDS,
    NEMO_ITN_LANGUAGES,
    NEMO_TN_LANGUAGES,
    NemoDependencyError,
    NemoFarError,
    NemoInverseNormalizerConfig,
    NemoLanguageNotSupportedError,
    NemoNormalizerConfig,
    clear_nemo_engine_cache,
    get_nemo_engine,
    get_nemo_itn_engine,
    initialize_nemo_engine,
    normalize_many_with_nemo,
    normalize_with_nemo,
    resolve_nemo_language,
    uses_official_nemo_itn,
    uses_official_nemo_tn,
    uses_private_ru_tn,
)

__all__ = [
    "DEFAULT_NEMO_CACHE_DIR",
    "NEMO_BACKENDS",
    "NEMO_ITN_LANGUAGES",
    "NEMO_TN_LANGUAGES",
    "NemoDependencyError",
    "NemoFarError",
    "NemoInverseNormalizerConfig",
    "NemoLanguageNotSupportedError",
    "NemoNormalizerConfig",
    "clear_nemo_engine_cache",
    "get_nemo_engine",
    "get_nemo_itn_engine",
    "initialize_nemo_engine",
    "normalize_many_with_nemo",
    "normalize_with_nemo",
    "resolve_nemo_language",
    "uses_official_nemo_itn",
    "uses_official_nemo_tn",
    "uses_private_ru_tn",
]
