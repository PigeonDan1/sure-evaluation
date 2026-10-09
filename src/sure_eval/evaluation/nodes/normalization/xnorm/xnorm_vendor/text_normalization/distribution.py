# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Capability boundary between the private full build and the public subset."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import FrozenSet, Iterable, Mapping, Optional


@dataclass(frozen=True)
class DistributionPolicy:
    """Public-build allowlist of languages, modes, and parameters."""

    public_build: bool
    languages: FrozenSet[str] = frozenset()
    modes: FrozenSet[str] = frozenset()
    api_parameters: FrozenSet[str] = frozenset()
    language_capabilities: Mapping[str, Mapping[str, str]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def private(cls) -> "DistributionPolicy":
        return cls(public_build=False)

    @classmethod
    def public(cls, capabilities: Mapping[str, object]) -> "DistributionPolicy":
        raw_matrix = capabilities.get("language_capabilities", {})
        frozen_matrix = MappingProxyType({
            language: MappingProxyType(dict(category_states))
            for language, category_states in raw_matrix.items()
        })
        return cls(
            public_build=True,
            languages=frozenset(capabilities.get("languages", ())),
            modes=frozenset(capabilities.get("modes", ())),
            api_parameters=frozenset(capabilities.get("api_parameters", ())),
            language_capabilities=frozen_matrix,
        )

    def validate_request(self, language: Optional[str], mode: Optional[str]) -> None:
        if not self.public_build:
            return
        if language not in self.languages:
            raise ValueError(f"Unsupported language in the public build: {language}")
        if mode not in self.modes:
            raise ValueError(f"Unsupported mode in the public build: {mode}")

    def validate_parameters(self, parameters: Iterable[str]) -> None:
        if not self.public_build:
            return
        unsupported = sorted(set(parameters) - self.api_parameters)
        if unsupported:
            raise ValueError(
                "Unsupported parameters in the public build: " + ", ".join(unsupported)
            )

    def filter_kwargs(
        self,
        kwargs: Mapping[str, object],
        explicit_parameters: Optional[Iterable[str]] = None,
    ) -> dict:
        if not self.public_build:
            return dict(kwargs)
        parameters = kwargs.keys() if explicit_parameters is None else explicit_parameters
        self.validate_parameters(parameters)
        return {
            key: value
            for key, value in kwargs.items()
            if key in self.api_parameters
        }

    def capabilities_for(self, language: str) -> Mapping[str, str]:
        if not self.public_build:
            return MappingProxyType({})
        if language not in self.languages:
            raise ValueError(f"Unsupported language in the public build: {language}")
        return self.language_capabilities[language]


def _load_distribution_policy() -> DistributionPolicy:
    """Public policy exists only in export artifacts; private source has full capabilities by default."""

    try:
        from ._public_build import PUBLIC_CAPABILITIES
    except ModuleNotFoundError as exc:
        expected_module = f"{__package__}._public_build"
        if exc.name != expected_module:
            raise
        return DistributionPolicy.private()
    return DistributionPolicy.public(PUBLIC_CAPABILITIES)


_POLICY = _load_distribution_policy()


def get_distribution_policy() -> DistributionPolicy:
    return _POLICY


def default_nemo_backend() -> str:
    """Private and public defaults are both direct; missing graphs fall back to official."""

    return "direct"


def get_language_capabilities(language: str) -> Mapping[str, str]:
    """Return the capability matrix when this build publishes one."""

    return _POLICY.capabilities_for(language)


def validate_distribution_request(language: Optional[str], mode: Optional[str]) -> None:
    _POLICY.validate_request(language, mode)


def filter_distribution_kwargs(
    kwargs: Mapping[str, object],
    explicit_parameters: Optional[Iterable[str]] = None,
) -> dict:
    return _POLICY.filter_kwargs(kwargs, explicit_parameters=explicit_parameters)
