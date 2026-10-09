# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

import re
from ..base import TextNormalization_Base
from ..logger import logger

alphabet_pattern = 'a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ' # Q, W, X, Y are not native Hungarian letters; loanwords only
diacritic_pattern = '' 

class TextNormalization_HU(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "hu"
        self.language_name_en = "Hungarian"
        self.language_name_zh = "匈牙利语"
        self.hello_world = "Helló, világ!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Other regexes need no override unless you know why

        # Default config values
        self.spaced_writing = True 
        self.score_method = "WER"  
        self.script = "Latin"         
        

