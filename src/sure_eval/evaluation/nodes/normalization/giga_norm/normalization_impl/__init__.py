"""Vendored GigaSpeechBench text normalization modules.

Upstream: https://github.com/SpeechColab/GigaSpeechBench (``text_norm/``).

Two deliberate deviations from upstream are applied so the code behaves as a
library instead of a script:

* Missing optional dependencies raise :class:`ImportError` instead of calling
  ``sys.exit(1)``.
* :func:`get_normalizer` re-raises :class:`ImportError` instead of silently
  falling back to the tag-stripping normalizer, so a broken environment can
  never be mistaken for a valid score.

The text normalization rules themselves are unmodified.
"""

from __future__ import annotations

import importlib
from typing import Callable

from ._common import remove_paralinguistic_tags

__all__ = [
    "PROFILE_ALIASES",
    "SUPPORTED_PROFILES",
    "get_normalizer",
    "remove_paralinguistic_tags",
    "resolve_profile",
]

# Profiles backed by a dedicated ``<CODE>.py`` module in this package.
SUPPORTED_PROFILES = (
    "AR",
    "ARE",
    "CHN",
    "DZA",
    "EGY",
    "IDN",
    "IRQ",
    "JPN",
    "KOR",
    "MAR",
    "MYS",
    "PHL",
    "SAU",
    "SYR",
    "THA",
    "USA",
    "VNM",
)

# Upstream ``text_norm._LANG_ALIASES``: dialect, accent, vertical-domain, and
# age-group codes that reuse an existing profile.
PROFILE_ALIASES = {
    # Chinese dialects reuse the CHN normalizer
    "GAN": "CHN",
    "JIN": "CHN",
    "MIN": "CHN",
    "WU": "CHN",
    "XIANG": "CHN",
    "YUE": "CHN",
    # Vertical-domain Chinese reuses CHN
    "AGR-CH": "CHN", "AIT-CH": "CHN", "ART-CH": "CHN", "BIO-CH": "CHN",
    "ECM-CH": "CHN", "EDU-CH": "CHN", "ENG-CH": "CHN", "ENT-CH": "CHN",
    "FIN-CH": "CHN", "HUM-CH": "CHN", "LAW-CH": "CHN", "MED-CH": "CHN",
    "MIL-CH": "CHN",
    # Vertical-domain English reuses USA
    "AGR-EN": "USA", "AIT-EN": "USA", "ART-EN": "USA", "BIO-EN": "USA",
    "ECM-EN": "USA", "EDU-EN": "USA", "ENG-EN": "USA", "ENT-EN": "USA",
    "FIN-EN": "USA", "HUM-EN": "USA", "LAW-EN": "USA", "MED-EN": "USA",
    "MIL-EN": "USA",
    # English accents reuse USA
    "CHN-EN": "USA", "IND-EN": "USA", "JPN-EN": "USA", "PHL-EN": "USA",
    "SCT-EN": "USA", "SGP-EN": "USA",
    # Older / children speech
    "CHILD-CH": "CHN", "OLD-CH": "CHN",
    "CHILD-EN": "USA", "OLD-EN": "USA",
    # Difficulty variants
    "JPN_HARD": "JPN",
    "KOR_HARD": "KOR",
}


def resolve_profile(profile: str) -> str:
    """Return the vendored module name backing ``profile``.

    Resolution order mirrors upstream ``text_norm.get_normalizer``: alias
    table, exact code, hyphen prefix, then underscore prefix.
    """

    if not isinstance(profile, str):
        raise TypeError("giga_norm profile must be a string, for example 'CHN'")

    code = profile.upper().strip()
    candidates: list[str] = []

    if code in PROFILE_ALIASES:
        candidates.append(PROFILE_ALIASES[code])
    if code in SUPPORTED_PROFILES:
        candidates.append(code)
    if "-" in code:
        candidates.append(code.split("-", 1)[0])
    if "_" in code:
        candidates.append(code.split("_", 1)[0])

    for candidate in candidates:
        if candidate in SUPPORTED_PROFILES:
            return candidate

    supported = ", ".join(SUPPORTED_PROFILES)
    aliases = ", ".join(sorted(PROFILE_ALIASES))
    raise ValueError(
        f"Unsupported giga_norm profile {profile!r}. "
        f"Supported profiles: {supported}. Supported aliases: {aliases}."
    )


def get_normalizer(profile: str) -> Callable[[str], str]:
    """Return the ``normalize`` callable for ``profile``.

    Raises :class:`ValueError` for unknown profiles and :class:`ImportError`
    when the profile's optional dependencies are not installed.
    """

    module_name = resolve_profile(profile)
    module = importlib.import_module(f"{__name__}.{module_name}")
    normalize = getattr(module, "normalize", None)
    if normalize is None:
        raise AttributeError(
            f"Vendored giga_norm module {module_name!r} does not define normalize()"
        )
    return normalize
