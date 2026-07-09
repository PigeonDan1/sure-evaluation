"""Standalone SURE evaluation package."""

__version__ = "0.1.0"

__all__ = [
    "Config",
    "configure_logging",
    "get_logger",
    "SUREEvaluator",
    "RPSManager",
]


def __getattr__(name: str):
    """Load standalone exports lazily."""
    if name == "Config":
        from sure_eval.core.config import Config

        return Config
    if name in {"configure_logging", "get_logger"}:
        from sure_eval.core.logging import configure_logging, get_logger

        return {"configure_logging": configure_logging, "get_logger": get_logger}[name]
    if name in {"SUREEvaluator", "RPSManager"}:
        from sure_eval.evaluation import RPSManager, SUREEvaluator

        return {"SUREEvaluator": SUREEvaluator, "RPSManager": RPSManager}[name]
    raise AttributeError(f"module 'sure_eval' has no attribute {name!r}")
