# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from ..base import TextNormalization_Base
from ..logger import logger
from ..numbers import number_to_words

# Alphabet
alphabet_pattern = '\u0e00-\u0e7f'
diacritic_pattern = ''

class TextNormalization_TH(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "th"
        self.language_name_en = "Thai"
        self.language_name_zh = "泰语"
        self.hello_world = "สวัสดีชาวโลก!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.spaced_writing = True # Spaced writing
        self.score_method = "CER"  # Score by character, not word
        self.script = "Thai"
        #self.keep_english_letters = False # this language has no English letters

    def classify_number_match(self, text, match):
        """Percent is spoken before the number. th is not a NeMo TN language, so no overlay."""
        end = match.end()
        if end >= len(text) or text[end] != "%":
            return None
        num = match.group().rstrip(".").replace(",", "")
        try:
            spoken = number_to_words(
                num, language=self.language, debug=bool(self.debug)
            )
        except Exception as exc:
            logger.warning(
                f"Thai percent conversion failed: num = {num}, reason: {exc}",
                exc_info=bool(self.debug),
            )
            return None
        if not spoken:
            return None
        return " เปอร์เซ็นต์" + spoken + " ", end + 1
