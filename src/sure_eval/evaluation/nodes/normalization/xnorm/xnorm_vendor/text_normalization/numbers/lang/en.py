# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""In-house English number grammar."""

from typing import List

from ..grouped import GroupWords, ThreeDigitGroupGrammar


SMALL_NUMBERS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
SCALES = (
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
)


class EnglishNumberGrammar(ThreeDigitGroupGrammar):
    language = "en"
    zero_word = SMALL_NUMBERS[0]
    minus_word = "minus"
    decimal_word = "point"
    digit_words = SMALL_NUMBERS
    scale_names = SCALES

    @staticmethod
    def _under_one_hundred(value: int) -> str:
        if value < 20:
            return SMALL_NUMBERS[value]
        tens, units = divmod(value, 10)
        return TENS[tens] if units == 0 else f"{TENS[tens]}-{SMALL_NUMBERS[units]}"

    @classmethod
    def under_one_thousand(cls, value: int) -> str:
        hundreds, remainder = divmod(value, 100)
        if hundreds == 0:
            return cls._under_one_hundred(remainder)
        result = f"{SMALL_NUMBERS[hundreds]} hundred"
        if remainder:
            result += f" and {cls._under_one_hundred(remainder)}"
        return result

    def join_groups(self, groups: List[GroupWords]) -> str:
        result = groups[0][2]
        for scale_index, group_value, words in groups[1:]:
            separator = (
                " and " if scale_index == 0 and group_value < 100 else ", "
            )
            result += separator + words
        return result
