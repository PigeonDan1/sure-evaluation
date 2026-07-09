from __future__ import annotations

import builtins
import importlib
import sys


def test_core_logging_imports_without_structlog(monkeypatch) -> None:
    original_logging_module = sys.modules.get("sure_eval.core.logging")
    original_structlog_module = sys.modules.get("structlog")
    sys.modules.pop("sure_eval.core.logging", None)
    sys.modules.pop("structlog", None)
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "structlog":
            raise ModuleNotFoundError("No module named 'structlog'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        module = importlib.import_module("sure_eval.core.logging")
        logger = module.get_logger("unit-test").bind(stage="fallback")

        logger.info("hello")

        assert module.structlog is None
    finally:
        sys.modules.pop("sure_eval.core.logging", None)
        if original_logging_module is not None:
            sys.modules["sure_eval.core.logging"] = original_logging_module
        if original_structlog_module is not None:
            sys.modules["structlog"] = original_structlog_module
