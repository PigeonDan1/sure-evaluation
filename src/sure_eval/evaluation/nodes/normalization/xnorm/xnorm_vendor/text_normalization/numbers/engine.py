# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Unified number-engine registry."""

from typing import Any, Dict

from ..logger import logger
from .families.east_slavic import RussianNumberGrammar
from .families.germanic import SwedishNumberGrammar
from .families.id_ms import IndonesianNumberGrammar
from .families.indic_standard import create_indic_standard_grammars
from .lang.ar import ArabicNumberGrammar
from .lang.de import GermanNumberGrammar
from .lang.en import EnglishNumberGrammar
from .lang.es import SpanishNumberGrammar
from .lang.fr import FrenchNumberGrammar
from .lang.he import HebrewNumberGrammar
from .lang.hu import HungarianNumberGrammar
from .lang.hy import ArmenianNumberGrammar
from .lang.it import ItalianNumberGrammar
from .lang.ja import JapaneseNumberGrammar
from .lang.ko import KoreanNumberGrammar
from .lang.pt import PortugueseNumberGrammar
from .lang.th import ThaiNumberGrammar
from .lang.vi import VietnameseNumberGrammar
from .lang.zh import ChineseNumberGrammar
from .protocol import NumberGrammar, parse_number_token


# Core grammars stay in this module. An optional extra registry may add more languages.
NATIVE_GRAMMARS: Dict[str, NumberGrammar] = {
    "ar": ArabicNumberGrammar(),
    "de": GermanNumberGrammar(),
    "en": EnglishNumberGrammar(),
    "es": SpanishNumberGrammar(),
    "fr": FrenchNumberGrammar(),
    "he": HebrewNumberGrammar(),
    "hu": HungarianNumberGrammar(),
    "hy": ArmenianNumberGrammar(),
    "id": IndonesianNumberGrammar(),
    "it": ItalianNumberGrammar(),
    "ja": JapaneseNumberGrammar(),
    "ko": KoreanNumberGrammar(),
    "pt": PortugueseNumberGrammar(),
    "ru": RussianNumberGrammar(),
    "sv": SwedishNumberGrammar(),
    "th": ThaiNumberGrammar(),
    "vi": VietnameseNumberGrammar(),
    "zh": ChineseNumberGrammar(),
}
NATIVE_GRAMMARS.update(create_indic_standard_grammars(("hi", "mr")))

# If the optional extra registry is missing, keep the first batch.
_private_registry_loaded = False
try:
    from .private_registry import PRIVATE_GRAMMARS
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}.private_registry":
        raise
else:
    NATIVE_GRAMMARS.update(PRIVATE_GRAMMARS)
    _private_registry_loaded = True

# Cantonese reuses Chinese when the extra registry is present.
if (
    _private_registry_loaded
    and "yue" not in NATIVE_GRAMMARS
    and "zh" in NATIVE_GRAMMARS
):
    NATIVE_GRAMMARS["yue"] = NATIVE_GRAMMARS["zh"]


def _resolve_grammar_language(language: str) -> str:
    # TTS variants share the base-language grammar; jp/ja_joyo are not aliases.
    if language.endswith("_tts"):
        return language[: -len("_tts")]
    return language


def has_native_number_grammar(language: str) -> bool:
    return _resolve_grammar_language(language) in NATIVE_GRAMMARS


def is_number_language_supported(language: str) -> bool:
    return has_native_number_grammar(language)


def number_to_words(
    value: Any,
    language: str,
    conversion: str = "cardinal",
    debug: bool = False,
) -> str:
    """Use in-house grammars only. Advanced conversions without their own impl fall back to cardinal."""

    grammar = NATIVE_GRAMMARS.get(_resolve_grammar_language(language))
    if grammar is None:
        raise NotImplementedError(f"language {language} has no number grammar")
    if conversion != "cardinal" and debug:
        logger.debug(
            f"number_conversion={conversion} fallback=simple_cardinal "
            f"language={language}"
        )
    if debug:
        logger.debug(f"number_provider=native_{language}")
    return grammar.cardinal(parse_number_token(value))


def digits_to_words(digits: str, language: str) -> str:
    grammar = NATIVE_GRAMMARS.get(_resolve_grammar_language(language))
    if grammar is None:
        raise NotImplementedError(f"language {language} has no digit grammar")
    return grammar.digit_sequence(digits)
