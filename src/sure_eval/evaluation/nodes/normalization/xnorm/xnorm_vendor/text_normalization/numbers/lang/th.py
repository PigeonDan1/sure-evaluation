# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Thai six-digit-group cardinal grammar。"""

from ..protocol import NumberToken


class ThaiNumberGrammar:
    language = "th"
    minus_word = "ติดลบ"
    decimal_word = "จุด"
    digit_words = (
        "ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด",
        "แปด", "เก้า",
    )
    _position_words = ("", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน")
    oracle_random_upper_bound = 10**12

    def _six_digit_group(self, value: int, padded: bool = False) -> str:
        digits = str(value)
        multiple_digits = padded or len(digits) > 1
        words = []
        for offset, char in enumerate(reversed(digits)):
            digit = int(char)
            if not digit:
                continue
            if offset == 0 and multiple_digits and digit == 1:
                word = "เอ็ด"
            elif offset == 1 and digit == 2:
                word = "ยี่"
            elif offset == 1 and digit == 1:
                word = ""
            else:
                word = self.digit_words[digit]
            words.append((offset, word + self._position_words[offset]))
        return "".join(word for _, word in reversed(words)) or self.digit_words[0]

    def _integer_to_words(self, value: int) -> str:
        if value < 10**6:
            return self._six_digit_group(value)
        groups = []
        remaining = value
        while remaining:
            remaining, group = divmod(remaining, 10**6)
            groups.append(group)
        result = self._six_digit_group(groups[-1])
        for group in reversed(groups[:-1]):
            result += "ล้าน"
            if group:
                result += self._six_digit_group(group, padded=True)
        return result

    def cardinal(self, token: NumberToken) -> str:
        result = self._integer_to_words(token.integer_value)
        fraction = token.significant_fraction_digits
        if fraction:
            result += self.decimal_word + "".join(
                self.digit_words[int(digit)] for digit in fraction
            )
        if token.negative and not token.is_zero:
            return self.minus_word + result
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)
