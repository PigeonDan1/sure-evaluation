# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Vietnamese provider-compatible cardinal grammar。"""

from ..protocol import NumberToken


class VietnameseNumberGrammar:
    language = "vi"
    minus_word = "âm"
    decimal_word = "phẩy"
    digit_words = (
        "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám",
        "chín",
    )
    _small = (
        "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám",
        "chín", "mười", "mười một", "mười hai", "mười ba", "mười bốn",
        "mười lăm", "mười sáu", "mười bảy", "mười tám", "mười chín",
    )
    _tens = {
        2: "hai mươi", 3: "ba mươi", 4: "bốn mươi", 5: "năm mươi",
        6: "sáu mươi", 7: "bảy mươi", 8: "tám mươi", 9: "chín mươi",
    }
    _scales = {
        1: "nghìn", 2: "triệu", 3: "tỷ", 4: "nghìn tỷ",
        5: "trăm nghìn tỷ", 6: "Quintillion", 7: "Sextillion",
        8: "Septillion", 9: "Octillion", 10: "Nonillion",
    }
    oracle_random_upper_bound = 10**12

    def _under_hundred(self, value: int) -> str:
        if value < 20:
            return self._small[value]
        tens, ones = divmod(value, 10)
        result = self._tens[tens]
        if ones:
            if ones == 1:
                unit = "mốt"
            elif ones == 5:
                unit = "lăm"
            else:
                unit = self.digit_words[ones]
            result += " " + unit
        return result

    def _under_thousand(self, value: int) -> str:
        if value < 100:
            return self._under_hundred(value)
        hundreds, remainder = divmod(value, 100)
        result = self.digit_words[hundreds] + " trăm"
        if 0 < remainder < 10:
            result += " lẻ " + self.digit_words[remainder]
        elif remainder:
            result += " " + self._under_hundred(remainder)
        return result

    def _integer_to_words(self, value: int) -> str:
        if value < 1000:
            return self._under_thousand(value)
        chunks = []
        remaining = value
        while remaining:
            remaining, chunk = divmod(remaining, 1000)
            chunks.append(chunk)
        words = []
        for scale_index in range(len(chunks) - 1, -1, -1):
            chunk = chunks[scale_index]
            if not chunk:
                continue
            words.append(self._under_thousand(chunk))
            if scale_index:
                words.append(self._scales[scale_index])
            lower_value = value % (1000**scale_index) if scale_index else 0
            if scale_index and 0 < lower_value < 100:
                words.append("lẻ")
        return " ".join(words)

    def cardinal(self, token: NumberToken) -> str:
        result = self._integer_to_words(token.integer_value)
        fraction = token.fraction_digits[:2].ljust(2, "0")
        if fraction and int(fraction):
            result += f" {self.decimal_word} " + self._integer_to_words(int(fraction))
        if token.negative and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)
