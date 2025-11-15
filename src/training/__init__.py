from .checkpoint import adapt_pretrained_key, extract_adapted_weights, safe_torch_load
from .ema import EMAHelper, SWAHelper
from .utils import (
    apply_residual_output,
    build_degradation_schedule,
    build_patch_schedule,
    damp_optimizer_moments,
    pick_device,
    resolve_step_value,
    schedule_index,
    set_worker_seed_base,
    unwrap_model,
    worker_seed_init,
)
__all__ = [
    "adapt_pretrained_key",
    "extract_adapted_weights",
    "safe_torch_load",
    "EMAHelper",
    "SWAHelper",
    "LossComputer",
    "to_y",
    "apply_residual_output",
    "build_degradation_schedule",
    "build_patch_schedule",
    "damp_optimizer_moments",
    "pick_device",
    "resolve_step_value",
    "schedule_index",
    "set_worker_seed_base",
    "unwrap_model",
    "worker_seed_init",
]


def __getattr__(name):
    if name in {"LossComputer", "to_y"}:
        from src.losses import LossComputer as _LossComputer, to_y as _to_y

        if name == "LossComputer":
            return _LossComputer
        return _to_y
    raise AttributeError(f"module 'src.training' has no attribute {name}")
