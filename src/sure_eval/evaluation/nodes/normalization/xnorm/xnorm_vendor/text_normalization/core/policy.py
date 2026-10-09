# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Central default policy for the three product modes."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ModePolicy:
    """Describe mode differences only; explicit language-class defaults win."""

    mode: str
    code_switch_tag: str
    remove_foreign_chars: bool

    def compile_config(
        self,
        options: Mapping[str, object],
        normalizer: Any,
    ) -> dict:
        config = dict(options)
        if config.get("code_switch_tag") is None:
            config["code_switch_tag"] = self.code_switch_tag
        if config.get("remove_foreign_chars") is None:
            language_default = getattr(normalizer, "remove_foreign_chars", None)
            # An explicit TTS alias False must not be overwritten by ordinary mode defaults.
            config["remove_foreign_chars"] = (
                self.remove_foreign_chars
                if language_default is None
                else language_default
            )
        return config


MODE_POLICIES = MappingProxyType({
    "asr_eval": ModePolicy(
        mode="asr_eval",
        code_switch_tag="delete",
        remove_foreign_chars=True,
    ),
    "asr_train": ModePolicy(
        mode="asr_train",
        code_switch_tag="keep_start_base",
        remove_foreign_chars=True,
    ),
    "tts": ModePolicy(
        mode="tts",
        code_switch_tag="keep_start_base",
        remove_foreign_chars=False,
    ),
})


def get_mode_policy(mode: str) -> ModePolicy:
    try:
        return MODE_POLICIES[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported normalization mode: {mode}") from exc
