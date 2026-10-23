# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Shared skeleton for three-digit grouped number grammars."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from .protocol import NumberToken


GroupWords = Tuple[int, int, str]


class ThreeDigitGroupGrammar(ABC):
    """Shared sign, three-digit split, scale, decimal, and digit-sequence flow."""

    language = ""
    zero_word = ""
    minus_word = ""
    decimal_word = ""
    digit_words: Tuple[str, ...] = ()
    fraction_digit_words: Tuple[str, ...] = ()
    sequence_digit_words: Tuple[str, ...] = ()
    scale_names: Tuple[str, ...] = ()
    digit_separator = " "

    @abstractmethod
    def under_one_thousand(self, value: int) -> str:
        """Language-specific readings for values in [1, 999]."""
        ...

    def format_scaled_group(
        self,
        scale_index: int,
        group_value: int,
        group_words: str,
        is_highest: bool,
    ) -> str:
        scale = self.scale_names[scale_index]
        return f"{group_words} {scale}".rstrip()

    def join_groups(self, groups: List[GroupWords]) -> str:
        return " ".join(words for _, _, words in groups)

    def integer_to_words(self, value: int) -> str:
        if value == 0:
            return self.zero_word

        raw_groups = []
        scale_index = 0
        while value:
            value, group_value = divmod(value, 1000)
            if group_value:
                if scale_index >= len(self.scale_names):
                    raise OverflowError(
                        f"{self.language} number exceeds supported scales"
                    )
                raw_groups.append((scale_index, group_value))
            scale_index += 1

        groups: List[GroupWords] = []
        highest_index = raw_groups[-1][0]
        for scale_index, group_value in reversed(raw_groups):
            group_words = self.under_one_thousand(group_value)
            groups.append(
                (
                    scale_index,
                    group_value,
                    self.format_scaled_group(
                        scale_index,
                        group_value,
                        group_words,
                        is_highest=scale_index == highest_index,
                    ),
                )
            )
        return self.join_groups(groups)

    def cardinal(self, token: NumberToken) -> str:
        result = self.integer_to_words(token.integer_value)
        fraction = token.significant_fraction_digits
        if fraction:
            digit_words = self.fraction_digit_words or self.digit_words
            result += f" {self.decimal_word} " + self.digit_separator.join(
                digit_words[int(digit)] for digit in fraction
            )
        if token.negative and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        digit_words = self.sequence_digit_words or self.digit_words
        return self.digit_separator.join(digit_words[int(digit)] for digit in digits)
