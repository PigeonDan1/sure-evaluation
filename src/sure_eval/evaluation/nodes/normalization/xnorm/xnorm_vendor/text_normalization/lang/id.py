# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from ..base import TextNormalization_Base

# Alphabet
alphabet_pattern = 'a-zA-Z'
diacritic_pattern = ''

class TextNormalization_ID(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "id"
        self.language_name_en = "Indonesian"
        self.language_name_zh = "印尼语"
        self.hello_world = "Halo, dunia!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.script = "Latin"
        
        
