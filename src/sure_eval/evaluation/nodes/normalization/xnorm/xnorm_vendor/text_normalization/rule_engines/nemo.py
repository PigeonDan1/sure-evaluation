# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Public NVIDIA NeMo Text Processing official TN/ITN adapter."""

from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
import fcntl
import io
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Type


# Keep the default cache under the repo so cwd changes do not create extra FAR copies.
DEFAULT_NEMO_CACHE_DIR = str(
    Path(__file__).resolve().parents[2] / ".cache" / "nemo"
)
USER_NEMO_CACHE_RELATIVE = Path(".cache") / "text_normalization" / "nemo"
NEMO_BACKENDS = frozenset({"official", "direct"})


@dataclass(frozen=True)
class NemoCacheLayout:
    """Writable dir plus read-only search paths. Write locks and .key files go only to write_dir."""

    write_dir: str
    read_dirs: Tuple[str, ...]


def user_nemo_cache_dir() -> Path:
    return Path.home() / USER_NEMO_CACHE_RELATIVE


def cache_directory_is_writable(path: Path) -> bool:
    probe = Path(path)
    while not probe.exists():
        if probe.parent == probe:
            return False
        probe = probe.parent
    return probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)


def cache_directory_is_readable(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and os.access(path, os.R_OK | os.X_OK)


def _same_cache_dir(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return os.path.normpath(str(left)) == os.path.normpath(str(right))


def resolve_nemo_cache_layout(
    explicit: Optional[str] = None,
    *,
    shared_dir: Optional[str] = None,
    user_dir: Optional[str] = None,
) -> NemoCacheLayout:
    """Explicit paths win; use the repo cache if writable, otherwise the user dir while read-only-reusing the repo."""

    if explicit:
        write = str(Path(explicit).expanduser())
        return NemoCacheLayout(write_dir=write, read_dirs=(write,))
    shared = Path(shared_dir or DEFAULT_NEMO_CACHE_DIR)
    if cache_directory_is_writable(shared):
        write = str(shared)
        return NemoCacheLayout(write_dir=write, read_dirs=(write,))
    user = Path(user_dir) if user_dir is not None else user_nemo_cache_dir()
    ordered = [str(user)]
    if cache_directory_is_readable(shared):
        ordered.append(str(shared))
    unique = tuple(dict.fromkeys(ordered))
    return NemoCacheLayout(write_dir=str(user), read_dirs=unique)

# This list is the official 1.2.0 TN packs wrapped by this project.
# The release includes rw, but 1.2.0 Normalizer raises KeyError: verbalize; this project does not wrap it.
NEMO_TN_LANGUAGES = frozenset(
    {
        "ar",
        "de",
        "en",
        "es",
        "fr",
        "hi",
        "hu",
        "hy",
        "it",
        "ja",
        "ko",
        "pt",
        "ru",
        "sv",
        "vi",
        "zh",
    }
)
NEMO_ITN_LANGUAGES = frozenset(
    {
        "ar",
        "de",
        "en",
        "es",
        "es_en",
        "fr",
        "he",
        "hi",
        "hi_en",
        "hy",
        "ja",
        "ko",
        "mr",
        "pt",
        "ru",
        "sv",
        "vi",
        "zh",
    }
)


def resolve_nemo_language(language: str) -> str:
    """TTS variants reuse the base language's official pack."""
    if language.endswith("_tts"):
        return language[: -len("_tts")]
    return language


# Official 1.2.0 Russian TN is non-deterministic only; this project pins deterministic=True and cannot autoload it.
NEMO_TN_DETERMINISTIC_UNSUPPORTED = frozenset({"ru"})


def uses_official_nemo_tn(language: str) -> bool:
    resolved = resolve_nemo_language(language)
    return (
        resolved in NEMO_TN_LANGUAGES
        and resolved not in NEMO_TN_DETERMINISTIC_UNSUPPORTED
    )


def uses_official_nemo_itn(language: str) -> bool:
    return resolve_nemo_language(language) in NEMO_ITN_LANGUAGES


def uses_private_ru_tn(language: str) -> bool:
    """When the private tree has a ru ClassifyFst pack, TN uses unique private readings instead of the official non-deterministic graph."""

    if resolve_nemo_language(language) != "ru":
        return False
    root = Path(__file__).resolve().parents[1]
    pack = root / "nemo_rules" / "ru" / "taggers" / "tokenize_and_classify.py"
    engine = Path(__file__).resolve().parent / "nemo_ru_tn.py"
    return pack.is_file() and engine.is_file()


class NemoDependencyError(RuntimeError):
    """NeMo, Pynini, or the direct module is unavailable."""


class NemoLanguageNotSupportedError(ValueError):
    """This pinned NeMo version does not support the language in that direction."""


class NemoFarError(RuntimeError):
    """direct FAR missing, malformed, or failed to compose."""


def unique_nemo_cache_tmp(cache_path: Path) -> Path:
    """Each process uses its own temporary FAR so a shared `.tmp.far` cannot be truncated concurrently."""

    token = f"{os.getpid()}.{secrets.token_hex(4)}"
    return cache_path.with_name(f"{cache_path.stem}.{token}.tmp.far")


def copy_cache_artifact(src: Path, dest: Path) -> None:
    """Copy into the writable dir via a unique tempfile then replace; never touch the read-only source."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = dest.with_name(
        f"{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        shutil.copyfile(src, temporary_path)
        temporary_path.replace(dest)
    except Exception:
        if temporary_path.is_file():
            temporary_path.unlink(missing_ok=True)
        raise


def language_cache_file_matches(name: str, language: str) -> bool:
    tokens = [language]
    if language == "ja":
        tokens.append("jp")
    for token in tokens:
        if (
            name == token
            or name.startswith(f"{token}_")
            or name.startswith(f"_{token}_")
            or f"_{token}_" in name
        ):
            return True
    return False


def iter_language_cache_files(cache_dir: Path, language: str) -> List[Path]:
    if not cache_dir.is_dir():
        return []
    files = []
    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(".lock") or name.endswith(".tmp") or ".tmp." in name:
            continue
        if language_cache_file_matches(name, language):
            files.append(path)
    return files


def adopt_readonly_cache(
    write_dir: Path,
    search_dirs: Sequence[str],
    sidecar_name: str,
    expected: str,
    language: str,
) -> bool:
    """If a read-only fingerprint matches, copy into write_dir. Return whether write_dir already has a matching sidecar."""

    write_dir = Path(write_dir)
    write_key = write_dir / sidecar_name
    if read_fingerprint_file(write_key) == expected:
        return True
    for raw in search_dirs:
        src_dir = Path(raw)
        if _same_cache_dir(src_dir, write_dir):
            continue
        src_key = src_dir / sidecar_name
        if read_fingerprint_file(src_key) != expected:
            continue
        for src in iter_language_cache_files(src_dir, language):
            copy_cache_artifact(src, write_dir / src.name)
        if src_key.is_file() and not write_key.is_file():
            copy_cache_artifact(src_key, write_key)
        return read_fingerprint_file(write_key) == expected
    return False



@contextmanager
def _suppress_official_stdio() -> Iterator[None]:
    """Some official graph_utils print Created FAR; do not let that pollute CLI stdout."""

    if _nemo_verbose():
        yield
        return
    with redirect_stdout(io.StringIO()):
        yield


@contextmanager
def locked_nemo_cache(lock_path: Path) -> Iterator[None]:
    """Cross-process exclusive compose. The kernel releases flock on crash; the lock file may remain."""

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(os.fspath(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        # LOCK_EX waits: later jobs read a finished FAR instead of compiling in parallel.
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def _maybe_locked_nemo_cache(lock_path: Optional[Path]) -> Iterator[None]:
    if lock_path is None:
        yield
        return
    with locked_nemo_cache(lock_path):
        yield


def _official_init_lock_path(
    cache_dir: Optional[str],
    language: str,
    direction: str,
    input_case: str,
) -> Optional[Path]:
    if not cache_dir or cache_dir == "None":
        return None
    return Path(cache_dir) / f"{language}_official_{direction}_{input_case}.lock"


NEMO_CACHE_PROTOCOL = 1


def nemo_dependency_versions() -> Dict[str, str]:
    """Use the released version as the official grammar fingerprint instead of hashing all of site-packages on every start."""

    versions = {}
    for name in ("nemo-text-processing", "pynini"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def hash_optional_file(path: Optional[str]) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def hash_source(obj: Any) -> str:
    """Composer function source enters the fingerprint; unrelated runtime lock edits must not bust FARs."""

    try:
        text = inspect.getsource(obj)
    except (OSError, TypeError):
        text = getattr(obj, "__qualname__", repr(obj))
        code = getattr(obj, "__code__", None)
        if code is not None:
            text += repr(code.co_code)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_tree(root: Path) -> str:
    if not root.is_dir():
        return ""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes([0]))
        digest.update(path.read_bytes())
        digest.update(bytes([0]))
    return digest.hexdigest()


def make_cache_fingerprint(parts: Mapping[str, str]) -> str:
    payload = {"protocol": str(NEMO_CACHE_PROTOCOL), **{key: str(value) for key, value in parts.items()}}
    canonical = "\n".join(f"{key}={payload[key]}" for key in sorted(payload))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_key_path(cache_path: Path) -> Path:
    return Path(cache_path).with_name(Path(cache_path).name + ".key")


def read_fingerprint_file(key_path: Path) -> Optional[str]:
    if not key_path.is_file():
        return None
    try:
        data = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("protocol") != NEMO_CACHE_PROTOCOL:
        return None
    fingerprint = data.get("fingerprint")
    return fingerprint if isinstance(fingerprint, str) and fingerprint else None


def write_fingerprint_file(key_path: Path, fingerprint: str) -> None:
    key_path = Path(key_path)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"protocol": NEMO_CACHE_PROTOCOL, "fingerprint": fingerprint},
        sort_keys=True,
    )
    temporary_path = key_path.with_name(
        f"{key_path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        temporary_path.write_text(payload + "\n", encoding="utf-8")
        temporary_path.replace(key_path)
    except Exception:
        if temporary_path.is_file():
            temporary_path.unlink(missing_ok=True)
        raise


def cache_needs_rebuild(
    stored: Optional[str],
    expected: str,
    overwrite: bool,
) -> bool:
    return bool(overwrite) or stored != expected


def official_cache_fingerprint(
    language: str,
    direction: str,
    input_case: str,
    whitelist: Optional[str] = None,
    deterministic: bool = True,
) -> str:
    parts = nemo_dependency_versions()
    parts.update(
        {
            "language": language,
            "direction": f"official_{direction}",
            "input_case": input_case,
            "deterministic": "1" if deterministic else "0",
            "whitelist": hash_optional_file(whitelist),
        }
    )
    return make_cache_fingerprint(parts)


def prepare_nemo_cache_file(
    cache_path: Path,
    overwrite: bool,
    loader: Callable[[Path], Any],
    builder: Callable[[Path], Any],
    fingerprint: Optional[str] = None,
    search_dirs: Optional[Sequence[str]] = None,
) -> Any:
    """Load on hit; if a read-only fingerprint matches, copy into the writable dir; otherwise lock and compose."""

    cache_path = Path(cache_path)
    key_path = cache_key_path(cache_path)
    readonly_sources = []
    for raw in search_dirs or ():
        src_dir = Path(raw)
        if _same_cache_dir(src_dir, cache_path.parent):
            continue
        readonly_sources.append(src_dir / cache_path.name)

    def is_valid(path: Path) -> bool:
        if overwrite or not path.is_file() or path.stat().st_size == 0:
            return False
        if fingerprint is not None and read_fingerprint_file(cache_key_path(path)) != fingerprint:
            return False
        return True

    def try_load(path: Path) -> Any:
        if not is_valid(path):
            return None
        try:
            return loader(path)
        except Exception:
            # Truncated or version-mismatched cache is a miss; the lock holder rebuilds it.
            return None

    loaded = try_load(cache_path)
    if loaded is not None:
        return loaded
    readonly_hit = None
    if fingerprint is not None:
        for src in readonly_sources:
            if is_valid(src):
                readonly_hit = src
                break
    with locked_nemo_cache(cache_path.with_name(cache_path.name + ".lock")):
        loaded = try_load(cache_path)
        if loaded is not None:
            return loaded
        if readonly_hit is not None and is_valid(readonly_hit):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            copy_cache_artifact(readonly_hit, cache_path)
            src_key = cache_key_path(readonly_hit)
            if src_key.is_file():
                copy_cache_artifact(src_key, key_path)
            loaded = try_load(cache_path)
            if loaded is not None:
                return loaded
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        built = builder(cache_path)
        if fingerprint is not None:
            write_fingerprint_file(key_path, fingerprint)
        return built


def call_with_cache_rebuild(
    factory: Callable[[bool], Any],
    overwrite_cache: bool,
) -> Any:
    """Official exists→load keeps failing on truncated FARs; force one rebuild when init fails."""

    try:
        return factory(overwrite_cache)
    except Exception:
        if overwrite_cache:
            raise
        return factory(True)


NEMO_LOGGER_NAME = "NeMo-text-processing"


class _NemoDebugFilter(logging.Filter):
    """Block official default INFO without editing NeMo. Official `normalize()` calls `logger.setLevel('INFO')`; filtering can only happen in the adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.debug = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if self.debug > 0:
            return True
        return record.levelno >= logging.WARNING


_NEMO_DEBUG_FILTER = _NemoDebugFilter()


def configure_nemo_logging(debug: int = 0) -> None:
    """Drive the official logger from `--debug`; INFO is off by default."""

    _NEMO_DEBUG_FILTER.debug = int(debug or 0)
    nemo_logger = logging.getLogger(NEMO_LOGGER_NAME)
    # Do not fall through to the repo-root DEBUG handler or official INFO is printed twice.
    nemo_logger.propagate = False
    if _NEMO_DEBUG_FILTER not in nemo_logger.filters:
        nemo_logger.addFilter(_NEMO_DEBUG_FILTER)
    nemo_logger.setLevel(
        logging.DEBUG if _NEMO_DEBUG_FILTER.debug > 0 else logging.WARNING
    )


def _nemo_verbose() -> bool:
    return _NEMO_DEBUG_FILTER.debug > 0


_FST_DUMP_RE = re.compile(
    r'tokens\s*\{|\b(?:name|integer|fraction|ordinal):\s*"'
)
_CJK_MINUS_ZERO_RE = re.compile("[\u3400-\u9fff]-0[\u3400-\u9fff]")


def _looks_like_corrupt_nemo_output(original: str, output: str) -> bool:
    """Official/direct sometimes return without raising, but write internal tags or illegal inserts."""

    if not isinstance(output, str):
        return True
    if _FST_DUMP_RE.search(output):
        return True
    # Corrupt CJK insert such as 需-0责 ("minus zero"). Block only when the source has no -0.
    if _CJK_MINUS_ZERO_RE.search(output) and "-0" not in original:
        return True
    return False


def _reject_corrupt_nemo_output(original: str, output: str) -> str:
    if _looks_like_corrupt_nemo_output(original, output):
        logging.getLogger(__name__).warning(
            "abnormal NeMo output, keeping original %r: %r",
            original,
            output if isinstance(output, str) else type(output).__name__,
        )
        return original
    return output


def _fst_call_or_original(text: str, producer):
    """A per-line runtime exception must not kill the process; official grammar cannot be patched, so the adapter swallows it and keeps the source.

    Init/missing-dependency errors still propagate. This only guards normalize() on an already loaded engine.
    Successful returns that still contain internal FST tags also keep the source.
    """

    try:
        output = producer()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "NeMo runtime failed, keeping original %r: %s: %s",
            text,
            type(exc).__name__,
            exc,
        )
        return text
    return _reject_corrupt_nemo_output(text, output)


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, not {type(text).__name__}")


def _validated_texts(texts: Sequence[str], batch_size: int) -> List[str]:
    values = list(texts)
    if not all(isinstance(text, str) for text in values):
        raise TypeError("texts may contain only str")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    return values


@dataclass(frozen=True)
class NemoNormalizerConfig:
    """Cacheable init parameters for the official NeMo Normalizer."""

    language: str
    input_case: str = "cased"
    deterministic: bool = True
    cache_dir: Optional[str] = DEFAULT_NEMO_CACHE_DIR
    overwrite_cache: bool = False
    whitelist: Optional[str] = None
    post_process: bool = True
    cache_search_dirs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.language not in NEMO_TN_LANGUAGES:
            supported = ", ".join(sorted(NEMO_TN_LANGUAGES))
            raise NemoLanguageNotSupportedError(
                f"NeMo TN 1.2.0 does not support language {self.language!r}; supported: {supported}"
            )
        if self.input_case not in {"cased", "lower_cased"}:
            raise ValueError("nemo_input_case must be cased or lower_cased")
        object.__setattr__(self, "cache_search_dirs", tuple(self.cache_search_dirs or ()))


@dataclass(frozen=True)
class NemoInverseNormalizerConfig:
    """Cacheable init parameters for official NeMo InverseNormalizer."""

    language: str
    input_case: str = "lower_cased"
    cache_dir: Optional[str] = DEFAULT_NEMO_CACHE_DIR
    overwrite_cache: bool = False
    whitelist: Optional[str] = None
    cache_search_dirs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.language not in NEMO_ITN_LANGUAGES:
            supported = ", ".join(sorted(NEMO_ITN_LANGUAGES))
            raise NemoLanguageNotSupportedError(
                f"NeMo ITN 1.2.0 does not support language {self.language!r}; supported: {supported}"
            )
        if self.input_case not in {"cased", "lower_cased"}:
            raise ValueError("nemo_itn_input_case must be cased or lower_cased")
        object.__setattr__(self, "cache_search_dirs", tuple(self.cache_search_dirs or ()))


def _load_class(module_name: str, class_name: str, direction: str) -> Type[Any]:
    """Import NeMo only for official languages or ITN; other paths do not take the heavy dependency."""

    # Install the log filter before import, otherwise ClassifyFst INFO prints first.
    configure_nemo_logging(_NEMO_DEBUG_FILTER.debug)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise NemoDependencyError(
            f"NeMo {direction} is unavailable; install requirements-nemo.txt; original error: {exc}"
        ) from exc
    normalizer_class = getattr(module, class_name, None)
    if normalizer_class is None:
        raise NemoDependencyError(
            f"NeMo {direction} module is missing class {class_name}"
        )
    return normalizer_class


def _load_normalizer_class() -> Type[Any]:
    return _load_class(
        "nemo_text_processing.text_normalization.normalize",
        "Normalizer",
        "TN",
    )


def _load_inverse_normalizer_class() -> Type[Any]:
    return _load_class(
        "nemo_text_processing.inverse_text_normalization.inverse_normalize",
        "InverseNormalizer",
        "ITN",
    )


class NemoTextNormalizationEngine:
    """Wrap the official written-to-spoken Normalizer."""

    def __init__(self, config: NemoNormalizerConfig):
        self.config = config
        normalizer_class = _load_normalizer_class()
        # Official Normalizer writes FARs in place; cold start per language must be serial.
        lock_path = _official_init_lock_path(
            config.cache_dir,
            config.language,
            "tn",
            config.input_case,
        )
        with _maybe_locked_nemo_cache(lock_path):
            def factory(overwrite: bool) -> Any:
                return normalizer_class(
                    input_case=config.input_case,
                    lang=config.language,
                    deterministic=config.deterministic,
                    cache_dir=config.cache_dir,
                    overwrite_cache=overwrite,
                    whitelist=config.whitelist,
                    post_process=config.post_process,
                )

            expected = official_cache_fingerprint(
                config.language,
                "tn",
                config.input_case,
                config.whitelist,
                config.deterministic,
            )
            key_path = None
            overwrite = config.overwrite_cache
            if config.cache_dir and config.cache_dir != "None":
                key_path = Path(config.cache_dir) / (
                    f"{config.language}_official_tn_{config.input_case}.key"
                )
                if not config.overwrite_cache:
                    adopt_readonly_cache(
                        Path(config.cache_dir),
                        config.cache_search_dirs,
                        key_path.name,
                        expected,
                        config.language,
                    )
                overwrite = cache_needs_rebuild(
                    read_fingerprint_file(key_path),
                    expected,
                    config.overwrite_cache,
                )
            with _suppress_official_stdio():
                self._normalizer = call_with_cache_rebuild(factory, overwrite)
            if key_path is not None:
                write_fingerprint_file(key_path, expected)

    def normalize(self, text: str) -> str:
        _validate_text(text)
        if not text:
            return text
        # verbose follows --debug; official code still logs INFO when verbose=False.
        return _fst_call_or_original(
            text,
            lambda: self._normalizer.normalize(
                text,
                verbose=_nemo_verbose(),
                punct_pre_process=False,
                punct_post_process=False,
            ),
        )

    def normalize_list(
        self,
        texts: Sequence[str],
        batch_size: int = 1,
        n_jobs: int = 1,
    ) -> List[str]:
        values = _validated_texts(texts, batch_size)
        if not values:
            return []
        try:
            results = self._normalizer.normalize_list(
                values,
                verbose=_nemo_verbose(),
                punct_pre_process=False,
                punct_post_process=False,
                batch_size=batch_size,
                n_jobs=n_jobs,
            )
        except Exception:
            # If a batch FST call fails, retry line by line so one row cannot kill the batch.
            return [self.normalize(text) for text in values]
        return [
            _reject_corrupt_nemo_output(src, dst)
            for src, dst in zip(values, results)
        ]


class NemoInverseTextNormalizationEngine:
    """Wrap the official spoken-to-written InverseNormalizer."""

    def __init__(self, config: NemoInverseNormalizerConfig):
        self.config = config
        normalizer_class = _load_inverse_normalizer_class()
        lock_path = _official_init_lock_path(
            config.cache_dir,
            config.language,
            "itn",
            config.input_case,
        )
        with _maybe_locked_nemo_cache(lock_path):
            def factory(overwrite: bool) -> Any:
                return normalizer_class(
                    input_case=config.input_case,
                    lang=config.language,
                    cache_dir=config.cache_dir,
                    overwrite_cache=overwrite,
                    whitelist=config.whitelist,
                )

            expected = official_cache_fingerprint(
                config.language,
                "itn",
                config.input_case,
                config.whitelist,
            )
            key_path = None
            overwrite = config.overwrite_cache
            if config.cache_dir and config.cache_dir != "None":
                key_path = Path(config.cache_dir) / (
                    f"{config.language}_official_itn_{config.input_case}.key"
                )
                if not config.overwrite_cache:
                    adopt_readonly_cache(
                        Path(config.cache_dir),
                        config.cache_search_dirs,
                        key_path.name,
                        expected,
                        config.language,
                    )
                overwrite = cache_needs_rebuild(
                    read_fingerprint_file(key_path),
                    expected,
                    config.overwrite_cache,
                )
            with _suppress_official_stdio():
                self._normalizer = call_with_cache_rebuild(factory, overwrite)
            if key_path is not None:
                write_fingerprint_file(key_path, expected)

    def normalize(self, text: str) -> str:
        _validate_text(text)
        if not text:
            return text
        return _fst_call_or_original(
            text,
            lambda: self._normalizer.inverse_normalize(text, verbose=_nemo_verbose()),
        )

    def normalize_list(
        self,
        texts: Sequence[str],
        batch_size: int = 1,
        n_jobs: int = 1,
    ) -> List[str]:
        values = _validated_texts(texts, batch_size)
        if not values:
            return []
        try:
            results = self._normalizer.normalize_list(
                values,
                verbose=_nemo_verbose(),
                punct_pre_process=False,
                punct_post_process=False,
                batch_size=batch_size,
                n_jobs=n_jobs,
            )
        except Exception:
            # If a batch FST call fails, retry line by line so one row cannot kill the batch.
            return [self.normalize(text) for text in values]
        return [
            _reject_corrupt_nemo_output(src, dst)
            for src, dst in zip(values, results)
        ]


@lru_cache(maxsize=32)
def get_nemo_engine(config: NemoNormalizerConfig) -> NemoTextNormalizationEngine:
    """Reuse official TN initialization by language and config."""

    return NemoTextNormalizationEngine(config)


@lru_cache(maxsize=32)
def get_nemo_itn_engine(
    config: NemoInverseNormalizerConfig,
) -> NemoInverseTextNormalizationEngine:
    """Reuse official ITN initialization by language and config."""

    return NemoInverseTextNormalizationEngine(config)


def _load_private_ru_tn_module() -> Any:
    module_name = f"{__package__}.nemo_ru_tn"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise NemoDependencyError(
            "this build does not include Russian TN"
        ) from exc


def _load_private_ru_direct_module() -> Any:
    module_name = f"{__package__}.nemo_ru_direct"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise NemoDependencyError(
            "this build does not include Russian direct TN"
        ) from exc


def _load_private_direct_module() -> Any:
    module_name = f"{__package__}.nemo_direct"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise NemoDependencyError(
            "direct backend is not available; use official"
        ) from exc


def _maybe_wrap_overlay_tn_engine(engine: Any, language: str, itn: bool) -> Any:
    """Wrap the TN engine with overlay rules when nemo_overlay is importable. ITN is not wrapped."""

    if itn or not hasattr(engine, "normalize"):
        return engine
    resolved = resolve_nemo_language(language)
    module_name = f"{__package__}.nemo_overlay"
    try:
        overlay_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        return engine
    return overlay_module.wrap_overlay_tn_engine(engine, resolved)


def clear_nemo_engine_cache() -> None:
    """Clear official caches; also clear direct/overlay caches if those modules are loaded."""

    get_nemo_engine.cache_clear()
    get_nemo_itn_engine.cache_clear()
    direct_module = sys.modules.get(f"{__package__}.nemo_direct")
    if direct_module is not None:
        direct_module.clear_nemo_direct_engine_cache()
    overlay_module = sys.modules.get(f"{__package__}.nemo_overlay")
    if overlay_module is not None:
        overlay_module.clear_nemo_overlay_cache()
    ru_module = sys.modules.get(f"{__package__}.nemo_ru_tn")
    if ru_module is not None:
        ru_module.clear_private_ru_tn_cache()
    ru_direct_module = sys.modules.get(f"{__package__}.nemo_ru_direct")
    if ru_direct_module is not None:
        ru_direct_module.clear_private_ru_direct_cache()


def initialize_nemo_engine(
    language: str,
    input_case: Optional[str] = None,
    cache_dir: Optional[str] = None,
    overwrite_cache: bool = False,
    whitelist: Optional[str] = None,
    itn: bool = False,
    backend: str = "official",
    debug: int = 0,
) -> Any:
    """Initialize the NeMo backend. direct only wraps DIRECT_* languages, otherwise falls back to official; languages outside NEMO_* raise. Overlay wraps TN only when nemo_overlay is importable."""

    configure_nemo_logging(debug)
    if backend not in NEMO_BACKENDS:
        raise ValueError("nemo_backend must be official or direct")
    layout = resolve_nemo_cache_layout(cache_dir)
    if not itn and uses_private_ru_tn(language):
        # Private ru is not in nemo_direct.py: official=whole-sentence ClassifyFst, direct=Python fast path.
        effective_input_case = input_case or "cased"
        config = NemoNormalizerConfig(
            language="ru",
            input_case=effective_input_case,
            cache_dir=layout.write_dir,
            overwrite_cache=overwrite_cache,
            whitelist=whitelist,
            cache_search_dirs=layout.read_dirs,
        )
        if backend == "direct":
            ru_direct = _load_private_ru_direct_module()
            return ru_direct.initialize_private_ru_direct_engine(config)
        ru_module = _load_private_ru_tn_module()
        return ru_module.initialize_private_ru_tn_engine(config)
    if backend == "direct":
        direct_module = _load_private_direct_module()
        if direct_module.supports_direct(language, itn):
            engine = direct_module.initialize_direct_nemo_engine(
                language=language,
                input_case=input_case,
                cache_dir=layout.write_dir,
                overwrite_cache=overwrite_cache,
                whitelist=whitelist,
                itn=itn,
                cache_search_dirs=layout.read_dirs,
            )
            return _maybe_wrap_overlay_tn_engine(engine, language, itn)

    effective_input_case = input_case or ("lower_cased" if itn else "cased")
    if itn:
        return get_nemo_itn_engine(
            NemoInverseNormalizerConfig(
                language=language,
                input_case=effective_input_case,
                cache_dir=layout.write_dir,
                overwrite_cache=overwrite_cache,
                whitelist=whitelist,
                cache_search_dirs=layout.read_dirs,
            )
        )
    config = NemoNormalizerConfig(
        language=language,
        input_case=effective_input_case,
        cache_dir=layout.write_dir,
        overwrite_cache=overwrite_cache,
        whitelist=whitelist,
        cache_search_dirs=layout.read_dirs,
    )
    return _maybe_wrap_overlay_tn_engine(get_nemo_engine(config), language, False)


def normalize_with_nemo(
    text: str,
    language: str,
    input_case: Optional[str] = None,
    cache_dir: Optional[str] = None,
    overwrite_cache: bool = False,
    whitelist: Optional[str] = None,
    itn: bool = False,
    backend: str = "official",
    debug: int = 0,
) -> str:
    """Reuse a NeMo TN/ITN engine with fixed parameters."""

    engine = initialize_nemo_engine(
        language=language,
        input_case=input_case,
        cache_dir=cache_dir,
        overwrite_cache=overwrite_cache,
        whitelist=whitelist,
        itn=itn,
        backend=backend,
        debug=debug,
    )
    return engine.normalize(text)


def normalize_many_with_nemo(
    texts: Sequence[str],
    language: str,
    input_case: Optional[str] = None,
    cache_dir: Optional[str] = None,
    overwrite_cache: bool = False,
    whitelist: Optional[str] = None,
    batch_size: int = 1,
    n_jobs: int = 1,
    itn: bool = False,
    backend: str = "official",
    debug: int = 0,
) -> List[str]:
    """Reuse the same NeMo engine for batch TN/ITN."""

    engine = initialize_nemo_engine(
        language=language,
        input_case=input_case,
        cache_dir=cache_dir,
        overwrite_cache=overwrite_cache,
        whitelist=whitelist,
        itn=itn,
        backend=backend,
        debug=debug,
    )
    return engine.normalize_list(
        texts,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )
