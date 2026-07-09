"""Logging configuration for SURE-EVAL."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

try:
    import structlog
except ModuleNotFoundError:
    structlog = None

try:
    from rich.console import Console
    from rich.logging import RichHandler
except ModuleNotFoundError:
    Console = None
    RichHandler = None

console = Console() if Console is not None else None

_LOGGING_KWARGS = {"exc_info", "stack_info", "stacklevel", "extra"}


class _FallbackBoundLogger:
    """Small structlog-compatible adapter for minimal evaluator runtimes."""

    def __init__(self, name: str | None = None, context: dict[str, Any] | None = None) -> None:
        self._logger = logging.getLogger(name)
        self._context = context or {}

    def bind(self, **kwargs: Any) -> "_FallbackBoundLogger":
        return _FallbackBoundLogger(self._logger.name, {**self._context, **kwargs})

    def new(self, **kwargs: Any) -> "_FallbackBoundLogger":
        return _FallbackBoundLogger(self._logger.name, kwargs)

    def unbind(self, *keys: str) -> "_FallbackBoundLogger":
        context = {key: value for key, value in self._context.items() if key not in keys}
        return _FallbackBoundLogger(self._logger.name, context)

    def try_unbind(self, *keys: str) -> "_FallbackBoundLogger":
        return self.unbind(*keys)

    def _log(self, level: int, event: str, *args: Any, **kwargs: Any) -> None:
        log_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in _LOGGING_KWARGS}
        context = {**self._context, **kwargs}
        message = str(event)
        if context:
            rendered_context = " ".join(f"{key}={value!r}" for key, value in sorted(context.items()))
            message = f"{message} {rendered_context}"
        self._logger.log(level, message, *args, **log_kwargs)

    def debug(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, *args, **kwargs)

    def info(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, event, *args, **kwargs)

    def warning(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, *args, **kwargs)

    warn = warning

    def error(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, *args, **kwargs)

    def critical(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, *args, **kwargs)

    def exception(self, event: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, event, *args, **kwargs)


def configure_logging(
    level: str = "INFO",
    format_type: str = "structured",
    log_file: str | None = None,
) -> None:
    """Configure logging for SURE-EVAL."""
    
    # Configure standard logging
    handlers: list[logging.Handler] = []
    if RichHandler is not None:
        handlers.append(
            RichHandler(
                console=console,
                rich_tracebacks=True,
                markup=True,
            )
        )
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=handlers,
    )

    if structlog is None:
        return
    
    # Configure structlog
    if format_type == "structured":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def get_logger(name: str | None = None) -> Any:
    """Get a logger instance."""
    if structlog is None:
        return _FallbackBoundLogger(name)
    return structlog.get_logger(name)
