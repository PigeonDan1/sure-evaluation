# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Shared input protocol for multilingual number grammars."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Protocol, Union


class UnsupportedNumberConversionError(NotImplementedError):
    """The current grammar does not support this numeric input."""


NumberInput = Union[int, float, Decimal, str]
NUMBER_LITERAL_PATTERN = re.compile(
    r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$"
)


@dataclass(frozen=True)
class NumberToken:
    """Language-agnostic decimal number structure."""

    source: str
    negative: bool
    integer_digits: str
    fraction_digits: str = ""

    @property
    def integer_value(self) -> int:
        return int(self.integer_digits)

    @property
    def significant_fraction_digits(self) -> str:
        """Strip trailing decimal zeros to match historic cardinal semantics."""
        return self.fraction_digits.rstrip("0")

    @property
    def is_zero(self) -> bool:
        return self.integer_value == 0 and not self.significant_fraction_digits


class NumberGrammar(Protocol):
    """Minimal number interface shared by all in-house languages."""

    language: str

    def cardinal(self, token: NumberToken) -> str:
        """Emit the ordinary cardinal reading."""
        ...

    def digit_sequence(self, digits: str) -> str:
        """Emit a reading for each digit."""
        ...


def parse_number_token(value: NumberInput) -> NumberToken:
    """Parse sign/integer/fraction and take on no linguistic rules."""

    if isinstance(value, bool):
        raise TypeError("bool is not a valid number TN input")
    source = str(value).strip()
    if "e" in source.lower():
        try:
            source = format(Decimal(source), "f")
        except InvalidOperation as exc:
            raise ValueError(f"invalid number: {value!r}") from exc
    if not NUMBER_LITERAL_PATTERN.fullmatch(source):
        raise ValueError(f"invalid number: {value!r}")

    source = source.replace(",", "")
    negative = source.startswith("-")
    unsigned = source.lstrip("+-")
    integer_digits, separator, fraction_digits = unsigned.partition(".")
    integer_digits = integer_digits.lstrip("0") or "0"
    return NumberToken(
        source=str(value),
        negative=negative,
        integer_digits=integer_digits,
        fraction_digits=fraction_digits if separator else "",
    )
