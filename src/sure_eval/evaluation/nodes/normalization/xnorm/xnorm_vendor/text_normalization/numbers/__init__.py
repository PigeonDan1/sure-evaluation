# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Project-wide number parsing and grammar interface."""

from .engine import (
    NATIVE_GRAMMARS,
    digits_to_words,
    has_native_number_grammar,
    is_number_language_supported,
    number_to_words,
)
from .spans import NumberSpanPolicy, rewrite_number_spans
from .protocol import (
    NumberGrammar,
    NumberInput,
    NumberToken,
    UnsupportedNumberConversionError,
    parse_number_token,
)

__all__ = [
    "NATIVE_GRAMMARS",
    "NumberGrammar",
    "NumberInput",
    "NumberToken",
    "UnsupportedNumberConversionError",
    "NumberSpanPolicy",
    "digits_to_words",
    "has_native_number_grammar",
    "is_number_language_supported",
    "number_to_words",
    "parse_number_token",
    "rewrite_number_spans",
]
