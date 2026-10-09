# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""In-house Chinese number grammar."""

from ..protocol import NumberToken


DIGITS = "零一二三四五六七八九"
SMALL_UNITS = ("", "十", "百", "千")
LARGE_UNITS = ("", "萬", "億", "兆", "京", "垓", "秭", "穣")


class ChineseNumberGrammar:
    language = "zh"

    @staticmethod
    def _group_to_words(value: int, omit_leading_one: bool) -> str:
        result = []
        pending_zero = False
        for unit_index in range(3, -1, -1):
            divisor = 10 ** unit_index
            digit = value // divisor % 10
            if digit == 0:
                if result and value % divisor:
                    pending_zero = True
                continue
            if pending_zero:
                result.append(DIGITS[0])
                pending_zero = False
            if not (
                omit_leading_one
                and digit == 1
                and unit_index == 1
                and not result
            ):
                result.append(DIGITS[digit])
            result.append(SMALL_UNITS[unit_index])
        return "".join(result)

    @classmethod
    def _integer_to_words(cls, value: int, negative: bool) -> str:
        if value == 0:
            return DIGITS[0]

        groups = []
        while value:
            value, group = divmod(value, 10000)
            groups.append(group)
        if len(groups) > len(LARGE_UNITS):
            raise OverflowError("Chinese number exceeds supported scales")

        result = []
        skipped_group = False
        highest_index = len(groups) - 1
        for group_index in range(highest_index, -1, -1):
            group = groups[group_index]
            if group == 0:
                if result:
                    skipped_group = True
                continue
            if result and (skipped_group or group < 1000) and result[-1] != DIGITS[0]:
                result.append(DIGITS[0])
            omit_leading_one = (
                not negative
                and not result
                and 10 <= group < 20
            )
            result.append(cls._group_to_words(group, omit_leading_one))
            result.append(LARGE_UNITS[group_index])
            skipped_group = False
        return "".join(result)

    def cardinal(self, token: NumberToken) -> str:
        result = self._integer_to_words(token.integer_value, token.negative)
        fraction = token.significant_fraction_digits
        if fraction:
            result += "點" + "".join(DIGITS[int(digit)] for digit in fraction)
        if token.negative and not token.is_zero:
            return "負" + result
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return "".join(DIGITS[int(digit)] for digit in digits)
