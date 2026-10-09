# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

import re
from ..base import TextNormalization_Base
from ..logger import logger

alphabet_pattern = '\\u0900-\\u097F' 
diacritic_pattern = '' 

class TextNormalization_MR(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "mr"
        self.language_name_en = "Marathi"
        self.language_name_zh = "马拉地语"
        self.hello_world = "नमस्कार, जग!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Other regexes need no override unless you know why

        # Default config values
        self.spaced_writing = True 
        self.score_method = "WER"  
        self.script = "Devanagari"         
        

