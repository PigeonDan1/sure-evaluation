"""Install a minimal DeepSpeed stub for metric-only inference.

Some metric images contain a partial DeepSpeed install that raises at import
time when CUDA_HOME is not set. The TTS/VC metric runners do not use DeepSpeed,
but Transformers and PyTorch Lightning import it opportunistically.
"""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec


def install_deepspeed_stub() -> None:
    """Register lightweight DeepSpeed modules in ``sys.modules``."""
    existing = sys.modules.get("deepspeed")
    if existing is not None and getattr(existing, "__sure_eval_stub__", False):
        return

    deepspeed = types.ModuleType("deepspeed")
    deepspeed.__version__ = "0.0.0"
    deepspeed.__sure_eval_stub__ = True
    deepspeed.__spec__ = ModuleSpec("deepspeed", loader=None, is_package=True)
    deepspeed.__path__ = []  # type: ignore[attr-defined]

    class DeepSpeedEngine:  # pragma: no cover - imported only by optional deps
        """Placeholder type for libraries that import DeepSpeed annotations."""

    def _unavailable(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("DeepSpeed is not available in the SURE metric runner")

    deepspeed.DeepSpeedEngine = DeepSpeedEngine
    deepspeed.init_distributed = _unavailable
    deepspeed.initialize = _unavailable

    ops = types.ModuleType("deepspeed.ops")
    ops.__spec__ = ModuleSpec("deepspeed.ops", loader=None, is_package=True)
    ops.__path__ = []  # type: ignore[attr-defined]
    adam = types.ModuleType("deepspeed.ops.adam")
    adam.__spec__ = ModuleSpec("deepspeed.ops.adam", loader=None)

    class _UnavailableDeepSpeedOptimizer:
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("DeepSpeed optimizers are not available in the SURE metric runner")

    class DeepSpeedCPUAdam(_UnavailableDeepSpeedOptimizer):
        pass

    class FusedAdam(_UnavailableDeepSpeedOptimizer):
        pass

    adam.DeepSpeedCPUAdam = DeepSpeedCPUAdam
    adam.FusedAdam = FusedAdam
    ops.adam = adam

    sys.modules["deepspeed"] = deepspeed
    sys.modules["deepspeed.ops"] = ops
    sys.modules["deepspeed.ops.adam"] = adam
