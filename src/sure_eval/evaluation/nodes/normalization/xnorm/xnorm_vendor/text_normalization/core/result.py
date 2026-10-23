# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Internal result/diagnostics that do not change default text output."""

from dataclasses import dataclass

from .diagnostics import NormalizationDiagnostics


@dataclass(frozen=True)
class NormalizationResult:
    text_norm: str
    diagnostics: NormalizationDiagnostics
