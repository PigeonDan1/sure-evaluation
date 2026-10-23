# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Build Hindi/Bengali Indian-scale grammar from versioned word lists."""

import json
from pathlib import Path

from ..protocol import NumberToken
from ..configured import ConfiguredRecursiveGrammar
from ..protocol import UnsupportedNumberConversionError


class IndicStandardGrammar:
    language = ""

    def __init__(self, language: str, config: dict):
        self.language = language
        self.under_hundred = tuple(config["under_hundred"])
        if len(self.under_hundred) != 100:
            raise ValueError(f"{language} under_hundred must contain 100 items")
        self.digit_words = self.under_hundred[:10]
        self.hundred_word = config["hundred_word"]
        self.hundred_joiner = config["hundred_joiner"]
        self.scales = tuple(
            (int(value), word) for value, word in config["scales"]
        )
        self.minus_word = config["minus_word"]
        self.decimal_word = config["decimal_word"]
        self.decimal_mode = config["decimal_mode"]
        self.oracle_random_upper_bound = 10**12

    def _integer_to_words(self, value: int) -> str:
        if value < 100:
            return self.under_hundred[value]
        for scale, word in self.scales:
            if value < scale:
                continue
            coefficient, remainder = divmod(value, scale)
            result = self._integer_to_words(coefficient) + " " + word
            if remainder:
                result += " " + self._integer_to_words(remainder)
            return result
        hundreds, remainder = divmod(value, 100)
        result = (
            self._integer_to_words(hundreds)
            + self.hundred_joiner
            + self.hundred_word
        )
        if remainder:
            result += " " + self.under_hundred[remainder]
        return result

    def cardinal(self, token: NumberToken) -> str:
        result = self._integer_to_words(token.integer_value)
        fraction = token.significant_fraction_digits
        if fraction:
            if self.decimal_mode == "digits_drop_leading_zero":
                fraction = str(int(fraction))
            result += f" {self.decimal_word} " + self.digit_sequence(fraction)
        if token.negative and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)


def create_indic_standard_grammars(languages=None) -> dict:
    path = Path(__file__).resolve().parent.parent / "data/indic_standard.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Indic standard data schema is invalid")
    selected = data["languages"]
    if languages is not None:
        wanted = set(languages)
        selected = {
            language: config
            for language, config in selected.items()
            if language in wanted
        }
    grammars = {
        language: IndicStandardGrammar(language, config)
        for language, config in selected.items()
        if "under_hundred" in config
    }
    grammars.update({
        language: IndicHybridGrammar(language, config)
        for language, config in selected.items()
        if "positive_under_thousand" in config and not config.get("positive_only")
    })
    grammars.update({
        language: IndicPositiveOnlyGrammar(language, config)
        for language, config in selected.items()
        if config.get("positive_only") and "tamil_prefixes" not in config
    })
    grammars.update({
        language: TamilPositiveOnlyGrammar(language, config)
        for language, config in selected.items()
        if "tamil_prefixes" in config
    })
    return grammars


class IndicHybridGrammar:
    """Explicitly combine Indic positive integers with light negative/decimal semantics."""

    def __init__(self, language: str, config: dict):
        self.language = language
        self.positive_under_thousand = tuple(config["positive_under_thousand"])
        self.positive_scales = tuple(
            (int(value), word) for value, word in config["positive_scales"]
        )
        self.light = ConfiguredRecursiveGrammar(language, config["light"])
        self.digit_words = tuple(self.positive_under_thousand[:10])
        self.oracle_random_upper_bound = 10**12

    def _positive_integer(self, value: int) -> str:
        digits = str(value)
        if len(digits) > 9:
            return self.digit_sequence(digits)
        groups = [digits[-3:]]
        prefix = digits[:-3]
        while prefix:
            groups.append(prefix[-2:])
            prefix = prefix[:-2]
        words = []
        for index in range(len(groups) - 1, -1, -1):
            group = int(groups[index])
            if not group:
                continue
            words.append(self.positive_under_thousand[group])
            if index:
                words.append(self.positive_scales[index - 1][1])
        return " ".join(words)

    def cardinal(self, token: NumberToken) -> str:
        if not token.negative and not token.fraction_digits and token.integer_value > 0:
            return self._positive_integer(token.integer_value)
        return self.light.cardinal(token)

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)


class IndicPositiveOnlyGrammar:
    """Migrate the Indic provider that currently accepts positive integers only; do not widen defaults."""

    def __init__(self, language: str, config: dict):
        self.language = language
        self.positive_under_thousand = tuple(config["positive_under_thousand"])
        self.positive_scales = tuple(
            (int(value), word) for value, word in config["positive_scales"]
        )
        self.digit_words = tuple(self.positive_under_thousand[:10])
        self.oracle_random_upper_bound = 10**12

    def _positive_integer(self, value: int) -> str:
        digits = str(value)
        if len(digits) > 9:
            return self.digit_sequence(digits)
        groups = [digits[-3:]]
        prefix = digits[:-3]
        while prefix:
            groups.append(prefix[-2:])
            prefix = prefix[:-2]
        words = []
        for index in range(len(groups) - 1, -1, -1):
            group = int(groups[index])
            if not group:
                continue
            words.append(self.positive_under_thousand[group])
            if index:
                words.append(self.positive_scales[index - 1][1])
        return " ".join(words)

    def cardinal(self, token: NumberToken) -> str:
        if token.negative or token.fraction_digits or token.integer_value == 0:
            raise UnsupportedNumberConversionError(
                f"{self.language} currently supports positive integers only"
            )
        return self._positive_integer(token.integer_value)

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)


class TamilPositiveOnlyGrammar(IndicPositiveOnlyGrammar):
    """Choose Tamil sandhi from versioned coefficient prefixes."""

    def __init__(self, language: str, config: dict):
        super().__init__(language, config)
        self.prefixes = {
            int(scale): {
                kind: tuple(values) for kind, values in forms.items()
            }
            for scale, forms in config["tamil_prefixes"].items()
        }

    def _positive_integer(self, value: int) -> str:
        digits = str(value)
        if len(digits) > 9:
            return self.digit_sequence(digits)
        words = []
        remainder = value
        for scale in (10**7, 10**5, 1000):
            coefficient, remainder = divmod(remainder, scale)
            if not coefficient:
                continue
            kind = "terminal" if remainder == 0 else "joined"
            words.append(self.prefixes[scale][kind][coefficient])
        if remainder:
            words.append(self.positive_under_thousand[remainder])
        return " ".join(words)
