# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Language-class factory, read-only profiles, and compiled-pipeline registry."""

from types import MappingProxyType
from typing import Any, Mapping, Type

from .pipeline import CompiledPipeline
from .profile import LocaleProfile
from .request import NormalizationRequest


class NormalizerRegistry:
    """Extract metadata at registration time; runtime requests always create a new normalizer."""

    def __init__(
        self,
        normalizer_types: Mapping[str, Type[Any]],
        metadata_instances: Mapping[str, Any],
    ):
        if set(normalizer_types) != set(metadata_instances):
            raise ValueError("normalizer types do not match metadata registry keys")
        self._normalizer_types = MappingProxyType(dict(normalizer_types))
        self._profiles = MappingProxyType({
            code: LocaleProfile.from_normalizer(code, metadata_instances[code])
            for code in sorted(normalizer_types)
        })

    @property
    def codes(self):
        return tuple(sorted(self._normalizer_types))

    def create(self, language: str):
        normalizer_type = self._normalizer_types.get(language)
        if normalizer_type is None:
            raise ValueError(f"Not supported language: {language}")
        return normalizer_type()

    def profile(self, language: str) -> LocaleProfile:
        try:
            return self._profiles[language]
        except KeyError as exc:
            raise ValueError(f"Not supported language: {language}") from exc

    def compile(self, request: NormalizationRequest) -> CompiledPipeline:
        normalizer = self.create(request.language)
        return CompiledPipeline.build(
            request=request,
            profile=self.profile(request.language),
            normalizer=normalizer,
        )
