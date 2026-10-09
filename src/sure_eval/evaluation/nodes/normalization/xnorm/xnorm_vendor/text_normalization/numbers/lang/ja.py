# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Default Japanese kanji cardinal grammar."""

from ..protocol import NumberToken


class JapaneseNumberGrammar:
    language = "ja"
    minus_word = "マイナス"
    decimal_word = "点"
    digit_words = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")
    _small_units = ((1000, "千"), (100, "百"), (10, "十"))
    _large_scales = (
        (10**48, "極"),
        (10**44, "載"),
        (10**40, "正"),
        (10**36, "澗"),
        (10**32, "溝"),
        (10**28, "穣"),
        (10**24, "秭"),
        (10**20, "垓"),
        (10**16, "京"),
        (10**12, "兆"),
        (10**8, "億"),
        (10**4, "万"),
    )
    oracle_random_upper_bound = 10**12

    def _under_ten_thousand(self, value: int) -> str:
        if value == 0:
            return self.digit_words[0]
        parts = []
        remainder = value
        for unit, word in self._small_units:
            coefficient, remainder = divmod(remainder, unit)
            if coefficient:
                if coefficient > 1:
                    parts.append(self.digit_words[coefficient])
                parts.append(word)
        if remainder:
            parts.append(self.digit_words[remainder])
        return "".join(parts)

    def _integer_to_words(self, value: int) -> str:
        if value < 10_000:
            return self._under_ten_thousand(value)
        for scale, word in self._large_scales:
            if value < scale:
                continue
            coefficient, remainder = divmod(value, scale)
            result = self._integer_to_words(coefficient) + word
            if remainder:
                result += self._integer_to_words(remainder)
            return result
        raise ValueError(f"ja cannot split number {value}")

    def cardinal(self, token: NumberToken) -> str:
        result = self._integer_to_words(token.integer_value)
        fraction = token.significant_fraction_digits
        if fraction:
            result += self.decimal_word + self.digit_sequence(fraction)
        if token.negative and not token.is_zero:
            return self.minus_word + result
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return "".join(self.digit_words[int(digit)] for digit in digits)
