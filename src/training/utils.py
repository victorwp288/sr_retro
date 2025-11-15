import random

import numpy as np
import torch
import torch.nn.functional as F

from src.data.degradations import DegradationConfig

_WORKER_BASE_SEED = 0


def set_worker_seed_base(seed):
    global _WORKER_BASE_SEED
    _WORKER_BASE_SEED = seed


def worker_seed_init(worker_id):
    worker_seed = (_WORKER_BASE_SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def resolve_step_value(raw, total_steps):
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value <= 1.0:
            return int(total_steps * value)
        return int(value)
    text = str(raw).strip().lower()
    if text.endswith("t"):
        try:
            frac = float(text[:-1])
        except ValueError:
            return 0
        return int(total_steps * frac)
    if text.endswith("%"):
        try:
            frac = float(text[:-1]) / 100.0
        except ValueError:
            return 0
        return int(total_steps * frac)
    try:
        value = float(text)
    except ValueError:
        return 0
    if value <= 1.0:
        return int(total_steps * value)
    return int(value)


def build_patch_schedule(data_cfg, total_steps):
    raw = data_cfg.get("patch_schedule") or []
    entries = []
    for item in raw:
        step = resolve_step_value(item.get("step", 0), total_steps)
        size = int(item.get("size", 0))
        if step < 0 or size <= 0:
            continue
        entries.append((step, size))
    entries.sort(key=lambda pair: pair[0])
    return entries


def build_degradation_schedule(base_cfg, schedule_cfg, total_steps):
    scheduled = []
    seen_zero = False
    for item in schedule_cfg or []:
        overrides = {k: item[k] for k in item if k != "step"}
        if not overrides:
            continue
        step = resolve_step_value(item.get("step", 0), total_steps)
        merged = dict(base_cfg)
        merged.update(overrides)
        scheduled.append((max(0, step), DegradationConfig(**merged)))
        if step == 0:
            seen_zero = True
    if not scheduled:
        return []
    if not seen_zero:
        scheduled.insert(0, (0, DegradationConfig(**base_cfg)))
    scheduled.sort(key=lambda pair: pair[0])
    return scheduled


def schedule_index(step, schedule):
    idx = 0
    for i, (boundary, _) in enumerate(schedule):
        if step >= boundary:
            idx = i
        else:
            break
    return idx


def apply_residual_output(sr, lr, scale, enabled):
    if not enabled or scale <= 1:
        return sr
    target_h = lr.shape[-2] * scale
    target_w = lr.shape[-1] * scale
    base = F.interpolate(lr, size=(target_h, target_w), mode="nearest")
    return sr + base


def damp_optimizer_moments(optimizer, factor):
    for state in optimizer.state.values():
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if isinstance(exp_avg, torch.Tensor):
            exp_avg.mul_(factor)
        if isinstance(exp_avg_sq, torch.Tensor):
            exp_avg_sq.mul_(factor)


def pick_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
