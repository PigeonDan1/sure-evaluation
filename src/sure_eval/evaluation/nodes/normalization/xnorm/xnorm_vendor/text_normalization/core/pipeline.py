# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Compiled normalization pipeline: configure once, reuse per line."""

from dataclasses import dataclass
import logging
from types import MappingProxyType
from typing import Any, Mapping

from .diagnostics import COUNTER_NAMES, NormalizationDiagnostics, PipelineMetrics
from .policy import ModePolicy, get_mode_policy
from .profile import LocaleProfile
from .request import NormalizationRequest
from .result import NormalizationResult
from ..distribution import default_nemo_backend


@dataclass(frozen=True)
class CompiledPipeline:
    """Freeze request/profile/policy; the legacy normalizer is only the wrapped backend."""

    request: NormalizationRequest
    profile: LocaleProfile
    policy: ModePolicy
    configuration: Mapping[str, object]
    _normalizer: Any

    @classmethod
    def build(
        cls,
        request: NormalizationRequest,
        profile: LocaleProfile,
        normalizer: Any,
    ) -> "CompiledPipeline":
        policy = get_mode_policy(request.mode)
        configuration = policy.compile_config(request.options, normalizer)
        for name, default in (
            ("itn", False),
            ("nemo_backend", default_nemo_backend()),
            ("nemo_input_case", "cased"),
            ("nemo_itn_input_case", "lower_cased"),
            ("nemo_overwrite_cache", False),
        ):
            if configuration.get(name) is None:
                configuration[name] = default
        normalizer.config(**configuration)
        return cls(
            request=request,
            profile=profile,
            policy=policy,
            configuration=MappingProxyType(dict(configuration)),
            _normalizer=normalizer,
        )

    def normalize(self, text: str) -> NormalizationResult:
        removed_before = self._normalizer.num_removed_lines
        counters_before = self._normalizer.runtime_counters.snapshot()
        self._normalizer.runtime_counters.increment("processed_lines")
        normalized_text = self._safe_pipeline(text)
        counters_after = self._normalizer.runtime_counters.snapshot()
        counter_deltas = {
            name: after - before
            for name, before, after in zip(
                COUNTER_NAMES,
                counters_before,
                counters_after,
            )
        }
        return NormalizationResult(
            text_norm=normalized_text,
            diagnostics=NormalizationDiagnostics(
                engine=self._normalizer.tn_engine,
                removed_line=self._normalizer.num_removed_lines > removed_before,
                **counter_deltas,
            ),
        )

    def normalize_text(self, text: str) -> str:
        # Default CLI does not consume per-line diagnostics, avoiding short-lived objects on million-line jobs.
        self._normalizer.runtime_counters.increment("processed_lines")
        return self._safe_pipeline(text)

    def _safe_pipeline(self, text: str) -> str:
        """A language-class pipeline override that raises must still not kill the whole CLI."""
        try:
            return self._normalizer.pipeline(text)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "pipeline failed, keeping original %r: %s: %s",
                text,
                type(exc).__name__,
                exc,
            )
            return text

    def metrics(self) -> PipelineMetrics:
        counters = self._normalizer.runtime_counters
        cache = self._normalizer.cached_num_map
        return PipelineMetrics(
            engine=self._normalizer.tn_engine,
            processed_lines=counters.processed_lines,
            removed_lines=self._normalizer.num_removed_lines,
            map_rule_hits=counters.map_rule_hits,
            number_matches=counters.number_matches,
            number_cache_hits=counters.number_cache_hits,
            number_cache_misses=counters.number_cache_misses,
            fallbacks=counters.fallbacks,
            number_cache_entries=len(cache),
            number_cache_capacity=cache.max_size,
        )
