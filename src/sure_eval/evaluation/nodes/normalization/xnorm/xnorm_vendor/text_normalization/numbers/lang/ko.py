# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Korean Sino-Korean cardinal grammar。"""

from ..protocol import NumberToken


class KoreanNumberGrammar:
    language = "ko"
    minus_word = "마이너스"
    decimal_word = "점"
    digit_words = ("영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
    _small_units = ((1000, "천"), (100, "백"), (10, "십"))
    _large_scales = (
        (10**68, "무량대수"), (10**64, "불가사의"), (10**60, "나유타"),
        (10**56, "아승기"), (10**52, "항하사"), (10**48, "극"),
        (10**44, "재"), (10**40, "정"), (10**36, "간"), (10**32, "구"),
        (10**28, "양"), (10**24, "자"), (10**20, "해"), (10**16, "경"),
        (10**12, "조"), (10**8, "억"), (10**4, "만"),
    )
    oracle_random_upper_bound = 10**12

    def _under_ten_thousand(self, value: int) -> str:
        if value == 0:
            return self.digit_words[0]
        words = []
        remainder = value
        for unit, unit_word in self._small_units:
            coefficient, remainder = divmod(remainder, unit)
            if coefficient:
                if coefficient > 1:
                    words.append(self.digit_words[coefficient])
                words.append(unit_word)
        if remainder:
            words.append(self.digit_words[remainder])
        return "".join(words)

    def _integer_to_words(self, value: int) -> str:
        if value < 10_000:
            return self._under_ten_thousand(value)
        for scale, unit_word in self._large_scales:
            if value < scale:
                continue
            coefficient, remainder = divmod(value, scale)
            if coefficient == 1 and scale == 10**4:
                result = unit_word
            else:
                result = self._integer_to_words(coefficient) + unit_word
            if remainder:
                result += " " + self._integer_to_words(remainder)
            return result
        raise ValueError(f"ko cannot split number {value}")

    def cardinal(self, token: NumberToken) -> str:
        result = self._integer_to_words(token.integer_value)
        fraction = token.significant_fraction_digits
        if fraction:
            result += f" {self.decimal_word} " + self.digit_sequence(fraction)
        if token.negative and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)
