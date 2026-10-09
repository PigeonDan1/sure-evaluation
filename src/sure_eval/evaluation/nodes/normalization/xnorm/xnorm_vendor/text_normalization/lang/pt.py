# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from ..base import TextNormalization_Base

# Alphabet
alphabet_pattern = 'a-zA-ZÀàÁáÂâÃãÉéÊêÍíÓóÔôÕõÚúÜüÇç'
diacritic_pattern = ''

class TextNormalization_PT(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "pt"
        self.language_name_en = "Portuguese"
        self.language_name_zh = "葡萄牙语"
        self.hello_world = "Olá, mundo!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.script = "Latin"
        
        
