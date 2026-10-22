# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from ..base import TextNormalization_Base
from ..logger import logger
from ..numbers import number_to_words

# Alphabet
alphabet_pattern = 'А-Яа-яЁё'
diacritic_pattern = ''

# Without an extra Russian TN pack, only rewrite years immediately followed by " год".
_RU_YEAR_LAST_GENITIVE = {
    "0": "",
    "1": "первого",
    "2": "второго",
    "3": "третьего",
    "4": "четвёртого",
    "5": "пятого",
    "6": "шестого",
    "7": "седьмого",
    "8": "восьмого",
    "9": "девятого",
}
_RU_YEAR_LAST_NOMINATIVE = {
    "0": "нулевой",
    "1": "первый",
    "2": "второй",
    "3": "третий",
    "4": "четвёртый",
    "5": "пятый",
    "6": "шестой",
    "7": "седьмой",
    "8": "восьмой",
    "9": "девятый",
}


class TextNormalization_RU(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "ru"
        self.language_name_en = "Russian"
        self.language_name_zh = "俄语"
        self.hello_world = "Привет, мир!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.script = "Cyrillic"
        #self.keep_english_letters = False # this language has no English letters

    def classify_number_match(self, text, match):
        """YYYY год: leftover years NeMo did not consume. Keep год itself in the source."""
        num = match.group().rstrip(".")
        if len(num) != 4 or not num.isdigit() or num[0] == "0":
            return None
        end = match.end()
        # " год" is exactly 4 characters; >= would treat a sentence-final year as bare.
        if end + 4 > len(text) or text[end:end + 4].lower() != " год":
            return None
        if end + 5 < len(text) and text[end + 5] not in (" ", "\n"):
            last_digit = _RU_YEAR_LAST_GENITIVE[num[3]]
        else:
            last_digit = _RU_YEAR_LAST_NOMINATIVE[num[3]]
        try:
            if num[0] == "1":
                head = number_to_words(
                    num[1:3] + "0", language=self.language, debug=bool(self.debug)
                )
                spoken = " тысяча " + head + " " + last_digit + " "
            else:
                head = number_to_words(
                    num[0:3] + "0", language=self.language, debug=bool(self.debug)
                )
                spoken = " " + head + " " + last_digit + " "
        except Exception as exc:
            logger.warning(
                f"Russian year conversion failed: num = {num}, reason: {exc}",
                exc_info=bool(self.debug),
            )
            return None
        # Include " год" in the replacement span so the cache key is not just 1994, which would year-read bare numbers.
        return spoken + text[end + 1:end + 4], end + 4
