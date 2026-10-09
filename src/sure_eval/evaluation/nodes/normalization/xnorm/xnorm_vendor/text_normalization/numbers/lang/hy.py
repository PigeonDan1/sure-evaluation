# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Armenian provider-compatible cardinal grammar。"""

from ..recursive import RecursiveScaleGrammar


class ArmenianNumberGrammar(RecursiveScaleGrammar):
    language = "hy"
    minus_word = "մինուս"
    decimal_word = "ամբողջ"
    digit_words = (
        "զրո", "մեկ", "երկու", "երեք", "չորս", "հինգ", "վեց", "յոթ",
        "ութ", "ինը",
    )
    number_words = (
        (1000, "հազար"),
        (100, "հարյուր"),
        (90, "իննսուն"),
        (80, "ութսուն"),
        (70, "յոթանասուն"),
        (60, "վաթսուն"),
        (50, "հիսուն"),
        (40, "քառասուն"),
        (30, "երեսուն"),
        (20, "քսան"),
        (19, "տասնինը"),
        (18, "տասնութ"),
        (17, "տասնյոթ"),
        (16, "տասնվեց"),
        (15, "տասնհինգ"),
        (14, "տասնչորս"),
        (13, "տասներեք"),
        (12, "տասներկու"),
        (11, "տասնմեկ"),
        (10, "տասը"),
        (9, "ինը"),
        (8, "ութ"),
        (7, "յոթ"),
        (6, "վեց"),
        (5, "հինգ"),
        (4, "չորս"),
        (3, "երեք"),
        (2, "երկու"),
        (1, "մեկ"),
        (0, "զրո"),
    )
    sequential_merge = True
    nested_split = True
    exact_lookup_max = 19
    enforce_max_value = False
    oracle_random_upper_bound = 10**12

    def integer_to_words(self, value: int) -> str:
        """Replay the old implementation's million-range and even-billion dispatch only."""

        if 10**6 <= value < 10**9:
            millions, remainder = divmod(value, 10**6)
            prefix = f"{self.integer_to_words(millions)} միլիոն"
            if remainder:
                prefix += " " + self.integer_to_words(remainder)
            return prefix
        if value == 10**9:
            return "մեկ միլիարդ"
        if 10**9 < value < 10**12 and value % 10**9 == 0:
            return f"{self.integer_to_words(value // 10**9)} միլիարդ"
        return self._compose_words(value).replace("հազար հազար", "միլիոն")

    def _merge_sequential_pair(self, left, right):
        """Keep the old provider's omitted-one, glued-tens, and historic grouping semantics."""

        left_text, left_value = left
        right_text, right_value = right
        if left_value == 1:
            if right_value == 1000:
                return right
            if right_value < 1000:
                return right
            left_text = self.digit_words[1]

        if right_value < left_value and 100 <= left_value < 1000:
            if right_value % 100 == 0:
                right_text = right_text[:-1] + "ի"
            return f"{left_text} {right_text}", left_value + right_value

        if right_value < 100:
            if left_value < 100:
                separator = " " if left_text == "իննսուն" else ""
                return (
                    left_text + separator + right_text,
                    left_value + right_value,
                )
            return f"{left_text} {right_text}", left_value + right_value

        return f"{left_text} {right_text}", left_value + right_value
