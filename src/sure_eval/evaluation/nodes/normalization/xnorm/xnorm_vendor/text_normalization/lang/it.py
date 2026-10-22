# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from ..base import TextNormalization_Base
from ..utils import simple_pattern_difference

# Alphabet
alphabet_pattern = 'a-zA-ZàÀèÈìÌòÒùÙéÉóÓ’' # Italian: 26 English letters + 7 accented letters + elision marks
diacritic_pattern = ''

class TextNormalization_IT(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "it"
        self.language_name_en = "Italian"
        self.language_name_zh = "意大利语"
        self.hello_world = "Ciao, mondo!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.script = "Latin"
        # Italian U+2019 can mark elision; do not subtract it from the alphabet as punctuation.
        self.marks_pattern, _ = simple_pattern_difference(self.marks_pattern, "’")
        
        
