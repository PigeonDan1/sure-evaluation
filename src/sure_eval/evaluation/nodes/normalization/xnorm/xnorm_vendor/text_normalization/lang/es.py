# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from ..base import TextNormalization_Base

# Alphabet
alphabet_pattern = 'a-zA-ZáÁéÉíÍóÓúÚêÊëËüÜãÃñÑ'
diacritic_pattern = ''

class TextNormalization_ES(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "es"
        self.language_name_en = "Spanish"
        self.language_name_zh = "西班牙语"
        self.hello_world = "¡Hola, mundo!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.script = "Latin"
        
        
