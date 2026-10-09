# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Stable Python API for new callers; the legacy file API remains exported from the package root."""

from collections.abc import Iterable, Iterator

from .core import CompiledPipeline, NormalizationResult


def compile_pipeline(language: str, **options) -> CompiledPipeline:
    """Configure once per language and mode for batch or in-service reuse."""

    # Lazily import the package-root factory to avoid a cycle during language registration.
    from . import create_compiled_pipeline

    return create_compiled_pipeline(language, **options)


def normalize(text: str, language: str, **options) -> NormalizationResult:
    """Normalize one string and return an immutable result plus diagnostics."""

    return compile_pipeline(language, **options).normalize(text)


def normalize_lines(
    lines: Iterable[str],
    language: str,
    **options,
) -> Iterator[NormalizationResult]:
    """Streamingly reuse one pipeline; callers own IO wrapping."""

    pipeline = compile_pipeline(language, **options)
    for line in lines:
        yield pipeline.normalize(line)
