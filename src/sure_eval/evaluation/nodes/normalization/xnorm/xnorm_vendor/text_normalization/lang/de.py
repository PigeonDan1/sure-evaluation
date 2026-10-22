# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

import re
from ..logger import logger
from ..base import TextNormalization_Base

# Alphabet
alphabet_pattern = 'a-zA-ZäöüßÄÖÜẞ'
diacritic_pattern = ''

class TextNormalization_DE(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "de"
        self.language_name_en = "German"
        self.language_name_zh = "德语"
        self.hello_world = "Hallo, Welt!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.script = "Latin"
        
    def fun_convert_case(self, text: str):
        """
        German lowercase 'ß' uppercases to two letters "SS", but "SS" lowercases to "ss", not back to 'ß'.
        To keep on-screen form stable, keep 'ß' as-is when uppercasing, and do not use uppercase "ẞ"
        """
        if self.case == "upper":
            text2 = ""
            for c in text:
                if c == 'ß':
                    text2 += c;
                else:
                    text2 += c.upper()
        elif self.case == "lower":
            text2 = text.lower()  # Lowercasing German here is expected
        else:
            text2 = text

        if self.debug > 0 and text != text2:
            logger.debug(f"before: {text}")
            logger.debug(f" after: {text2}")
        return text2    
