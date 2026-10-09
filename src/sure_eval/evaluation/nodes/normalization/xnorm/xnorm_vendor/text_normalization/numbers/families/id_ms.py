# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Austronesian number grammar shared by Indonesian and Malay."""

from typing import Tuple

from ..grouped import ThreeDigitGroupGrammar


INDONESIAN_DIGITS = (
    "nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh",
    "delapan", "sembilan",
)
MALAY_DIGITS = (
    "kosong", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh",
    "lapan", "sembilan",
)
MALAY_CARDINAL_DIGITS = ("nol", *MALAY_DIGITS[1:])
INDONESIAN_SCALES = (
    "", "ribu", "juta", "miliar", "triliun", "kuadriliun", "kuantiliun",
    "sekstiliun", "septiliun", "oktiliun", "noniliun", "desiliun",
)
MALAY_SCALES = (
    "", "ribu", "juta", "bilion", "triliun", "kuadriliun", "kuantiliun",
    "sekstiliun", "septiliun", "oktiliun", "noniliun", "desiliun",
)


class AustronesianNumberGrammar(ThreeDigitGroupGrammar):
    """ID/MS share construction order; stable differences live in word lists."""

    minus_word = "min"
    decimal_word = "koma"

    def under_one_thousand(self, value: int) -> str:
        if value < 10:
            return self.digit_words[value]
        if value == 10:
            return "sepuluh"
        if value == 11:
            return "sebelas"
        if value < 20:
            return f"{self.digit_words[value - 10]} belas"
        if value < 100:
            tens, units = divmod(value, 10)
            result = f"{self.digit_words[tens]} puluh"
            return result if units == 0 else f"{result} {self.digit_words[units]}"
        if value < 200:
            remainder = value - 100
            return "seratus" if remainder == 0 else f"seratus {self.under_one_thousand(remainder)}"
        hundreds, remainder = divmod(value, 100)
        result = f"{self.digit_words[hundreds]} ratus"
        return result if remainder == 0 else f"{result} {self.under_one_thousand(remainder)}"

    def format_scaled_group(
        self,
        scale_index: int,
        group_value: int,
        group_words: str,
        is_highest: bool,
    ) -> str:
        if scale_index == 1 and group_value == 1 and is_highest:
            return "seribu"
        return super().format_scaled_group(
            scale_index,
            group_value,
            group_words,
            is_highest,
        )


class IndonesianNumberGrammar(AustronesianNumberGrammar):
    language = "id"
    zero_word = "nol"
    digit_words: Tuple[str, ...] = INDONESIAN_DIGITS
    scale_names = INDONESIAN_SCALES


class MalayNumberGrammar(AustronesianNumberGrammar):
    language = "ms"
    # Plain cardinals keep the old ID-fallback nol; digit sequences use local kosong.
    zero_word = "nol"
    digit_words: Tuple[str, ...] = MALAY_CARDINAL_DIGITS
    sequence_digit_words: Tuple[str, ...] = MALAY_DIGITS
    scale_names = MALAY_SCALES
