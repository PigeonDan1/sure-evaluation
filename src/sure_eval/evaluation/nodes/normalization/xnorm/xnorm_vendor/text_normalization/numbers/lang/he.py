# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Hebrew provider-compatible default feminine cardinal grammar。"""

from ..protocol import NumberToken


ZERO = "אפס"
ONES = {
    1: ("אחת", "אחד", "אחת", "אחד"),
    2: ("שתיים", "שניים", "שתי", "שני"),
    3: ("שלוש", "שלושה", "שלוש", "שלושת"),
    4: ("ארבע", "ארבעה", "ארבע", "ארבעת"),
    5: ("חמש", "חמישה", "חמש", "חמשת"),
    6: ("שש", "שישה", "שש", "ששת"),
    7: ("שבע", "שבעה", "שבע", "שבעת"),
    8: ("שמונה", "שמונה", "שמונה", "שמונת"),
    9: ("תשע", "תשעה", "תשע", "תשעת"),
}
TEEN_TENS = (("עשר", "עשרה"), ("עשרה", "עשר"), ("שתים עשרה", "שנים עשר"))
TENS = ("", "", "עשרים", "שלושים", "ארבעים", "חמישים", "שישים", "שבעים", "שמונים", "תשעים")
HUNDREDS = ("", "מאה", "מאתיים")
LARGE = (
    "", "מיליון", "מיליארד", "טריליון", "קוודריליון", "קווינטיליון",
    "סקסטיליון", "ספטיליון", "אוקטיליון", "נוניליון", "דסיליון",
    "אונדסיליון", "דואודסיליון", "טרדסיליון", "קווטואורדסיליון",
    "קווינדסיליון", "סקסדסיליון", "ספטנדסיליון", "אוקטודסיליון",
    "נובמדסיליון", "ויגינטיליון",
)


class HebrewNumberGrammar:
    language = "he"
    minus_word = "מינוס"
    decimal_word = "נקודה"
    digit_words = (ZERO, *(ONES[value][0] for value in range(1, 10)))
    max_value = 10**66
    oracle_random_upper_bound = 10**12

    def _chunk_words(self, group_index: int, chunk: int) -> list[str]:
        ones, tens, hundreds = chunk % 10, chunk // 10 % 10, chunk // 100
        words = []
        if hundreds:
            words.append(
                HUNDREDS[hundreds]
                if hundreds <= 2
                else f"{ONES[hundreds][0]} מאות"
            )
        if tens > 1:
            words.append(TENS[tens])

        if group_index == 0 or chunk >= 11:
            masculine = int(group_index > 0)
            if tens == 1:
                if ones == 0:
                    words.append(TEEN_TENS[0][masculine])
                elif ones == 2:
                    words.append(TEEN_TENS[2][masculine])
                else:
                    words.append(
                        f"{ONES[ones][masculine]} {TEEN_TENS[1][masculine]}"
                    )
            elif ones:
                words.append(ONES[ones][masculine])

        if group_index == 1:
            if chunk >= 11:
                words[-1] += " אלף"
            elif ones == 0:
                words.append("עשרת אלפים")
            elif ones <= 2:
                words.append("אלף" if ones == 1 else "אלפיים")
            else:
                words.append(f"{ONES[ones][3]} אלפים")
        elif group_index > 1:
            scale = LARGE[group_index - 1]
            if chunk >= 11:
                words[-1] += " " + scale
            elif ones == 0:
                words.append("עשרה " + scale)
            elif ones == 1:
                words.append(scale)
            else:
                form = 3 if chunk == 2 else 1
                words.append(f"{ONES[ones][form]} {scale}")
        return words

    def _integer_to_words(self, value: int) -> str:
        if value == 0:
            return ZERO
        if value >= self.max_value:
            raise OverflowError("Hebrew number is too large")
        chunks = []
        remaining = value
        while remaining:
            chunks.append(remaining % 1000)
            remaining //= 1000
        words = []
        for group_index in range(len(chunks) - 1, -1, -1):
            chunk = chunks[group_index]
            if not chunk:
                continue
            words.extend(self._chunk_words(group_index, chunk))
            if len(words) > 1:
                words[-1] = "ו" + words[-1]
        return " ".join(words)

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
