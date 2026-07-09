"""Reusable SA-ASR cpWER conversion profile.

The task pipeline imports these profile-local conversion modules directly. This
file is the stable profile entry point for humans and agents that need to
inspect or reuse the conversion behavior.
"""

from __future__ import annotations

from sure_eval.evaluation.conversion.sa_asr__cpwer.stm_to_txt import convert_stm_to_txt
from sure_eval.evaluation.conversion.sa_asr__cpwer.txt_to_stm import convert_txt_to_stm

__all__ = ["convert_stm_to_txt", "convert_txt_to_stm"]
