import argparse
import inspect
import random
from contextlib import nullcontext
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


from src.utils.seed import set_seed
from src.models import EDSR
from src.data import PixelArtDataset, DegradationConfig
from src.data.utils import auto_split_paths, read_split_file
from torchmetrics.functional.image import structural_similarity_index_measure

_WORKER_BASE_SEED = 0
_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters

def _safe_torch_load(path, *, weights_only=False):
    load_kwargs = {"map_location": "cpu"}
    if weights_only and _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def set_worker_seed_base(seed):
    global _WORKER_BASE_SEED
    _WORKER_BASE_SEED = seed


def worker_seed_init(worker_id):
    worker_seed = (_WORKER_BASE_SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def _adapt_pretrained_key(key):
    for pref in ("module.", "model.", "_orig_mod."):
        if key.startswith(pref):
            key = key[len(pref):]

    if key.startswith("sub_mean") or key.startswith("add_mean"):
        return None

    if key.startswith("head.0."):
        return "head." + key[len("head.0."):]

    if key.startswith("body."):
        parts = key.split(".")
        if len(parts) >= 4:
            block_idx = parts[1]
            layer = parts[2]
            remainder = ".".join(parts[3:])
            if layer == "0":
                base = f"body.{block_idx}.conv1"
                return f"{base}.{remainder}" if remainder else base
            if layer == "2":
                base = f"body.{block_idx}.conv2"
                return f"{base}.{remainder}" if remainder else base
            return key

    if key.startswith("body_conv"):
        return key

    if key.startswith("tail.0.0."):
        return "tail.0." + key[len("tail.0.0."):]
    if key.startswith("tail.1."):
        return "tail.2." + key[len("tail.1."):]

    if key.startswith("tail.") or key.startswith("head."):
        return key

    return key


def _extract_adapted_weights(state_dict, model_state, allow_tail=True):
    adapted = {}
    matched_tail = set()
    for raw_key, tensor in state_dict.items():
        candidate_keys = []
        if raw_key in model_state:
            candidate_keys.append(raw_key)
        new_key = _adapt_pretrained_key(raw_key)
        if new_key is not None and new_key not in candidate_keys:
            candidate_keys.append(new_key)
        for key in candidate_keys:
            is_tail = key.startswith("tail.")
            if is_tail:
                if not allow_tail:
                    continue
            if key not in model_state:
                continue
            if model_state[key].shape != tensor.shape:
                continue
            adapted[key] = tensor
            if is_tail:
                matched_tail.add(key)
            break
    return adapted, matched_tail


def _to_y(t):
    if t.size(1) == 1:
        return t
    if t.size(1) != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {t.size(1)}")
    r = t[:, 0:1]
    g = t[:, 1:2]
    b = t[:, 2:3]
    return 0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0


def charbonnier_loss(sr, hr, eps):
    diff = sr - hr
    return torch.sqrt(diff * diff + eps * eps).mean()


def gradient_loss(sr, hr, kernel_x, kernel_y):
    sr_y = _to_y(sr)
    hr_y = _to_y(hr)
    kernel_x = kernel_x.to(sr_y.dtype)
    kernel_y = kernel_y.to(sr_y.dtype)
    grad_sr_x = F.conv2d(sr_y, kernel_x, padding=1)
    grad_hr_x = F.conv2d(hr_y, kernel_x, padding=1)
    grad_sr_y = F.conv2d(sr_y, kernel_y, padding=1)
    grad_hr_y = F.conv2d(hr_y, kernel_y, padding=1)
    diff_x = torch.abs(grad_sr_x - grad_hr_x).mean()
    diff_y = torch.abs(grad_sr_y - grad_hr_y).mean()
    return diff_x + diff_y


def build_loss_fn(config, device):
    loss_cfg = config.get("loss", {})
    weights = {
        "charbonnier": float(loss_cfg.get("charbonnier_weight", 0.0)),
        "l1": float(loss_cfg.get("l1_weight", 0.0)),
        "grad": float(loss_cfg.get("grad_weight", 0.0)),
        "lpips": float(loss_cfg.get("lpips_weight", 0.0)),
    }
    eps = float(loss_cfg.get("charbonnier_eps", 1e-3))
    lpips_model = None
    grad_kernel_x = None
    grad_kernel_y = None
    if weights["grad"] > 0.0:
        base = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)
        grad_kernel_x = base
        grad_kernel_y = base.transpose(2, 3)
    if weights["lpips"] > 0.0:
        import lpips

        lpips_model = lpips.LPIPS(net=loss_cfg.get("lpips_net", "alex")).to(device)
        lpips_model.eval()
        for param in lpips_model.parameters():
            param.requires_grad = False

    if not any(weights.values()):
        return lambda sr, hr: F.mse_loss(sr, hr)

    def compute(sr, hr):
        total = 0.0
        if weights["charbonnier"] > 0.0:
            total += weights["charbonnier"] * charbonnier_loss(sr, hr, eps)
        if weights["l1"] > 0.0:
            total += weights["l1"] * torch.abs(sr - hr).mean()
        if weights["grad"] > 0.0:
            total += weights["grad"] * gradient_loss(sr, hr, grad_kernel_x, grad_kernel_y)
        if weights["lpips"] > 0.0 and lpips_model is not None:
            sr_lp = sr.float().mul(2.0).sub(1.0)
            hr_lp = hr.float().mul(2.0).sub(1.0)
            lpips_value = lpips_model(sr_lp, hr_lp).mean()
            total += weights["lpips"] * lpips_value
        return total

    return compute


def _build_patch_schedule(data_cfg):
    raw = data_cfg.get("patch_schedule") or []
    entries = []
    for item in raw:
        step = int(item.get("step", 0))
        size = int(item.get("size", 0))
        if step < 0 or size <= 0:
            continue
        entries.append((step, size))
    entries.sort(key=lambda pair: pair[0])
    return entries


def _patch_index_for_step(step, schedule):
    idx = 0
    for i, (boundary, _) in enumerate(schedule):
        if step >= boundary:
            idx = i
        else:
            break
    return idx


def pick_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Enable pilot mode by capping training steps for quick sweeps.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    resume_state = None
    start_step = 0
    best_psnr = None
    pilot_mode = bool(args.pilot)

    training_cfg = config["training"]
    tail_only = bool(training_cfg.get("tail_only", True))
    early_stop_patience = int(training_cfg.get("early_stop_patience", 0) or 0)
    early_stop_min_delta = float(training_cfg.get("early_stop_min_delta", 0.0))
    seed = config.get("seed", 42)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    set_worker_seed_base(seed)

    device = pick_device()
    print(f"training on {device}")
    model_cfg = config["model"]
    model = EDSR(
        scale=model_cfg["scale"],
        n_feats=model_cfg["n_feats"],
        n_resblocks=model_cfg["n_resblocks"],
        res_scale=model_cfg["res_scale"],
    ).to(device)

    degradation_cfg = DegradationConfig(**config["degradations"])

    data_cfg = config["data"]
    patch_schedule = _build_patch_schedule(data_cfg)
    patch_ref = None
    patch_cursor = 0
    patch_manager = None
    if patch_schedule:
        patch_manager = Manager()
        patch_ref = patch_manager.Value("i", patch_schedule[0][1])

    auto_root = data_cfg.get("root")
    ratios = data_cfg.get("auto_split_ratio", {"train": 0.8, "val": 0.1, "test": 0.1})
    split_seed = data_cfg.get("auto_split_seed", seed)
    if auto_root:
        splits = auto_split_paths(auto_root, ratios, split_seed)
        train_paths = splits["train"]
        val_paths = splits["val"]
    else:
        train_source = data_cfg.get("train_split")
        val_source = data_cfg.get("val_split")
        if not train_source or not val_source:
            raise RuntimeError("train_split and val_split must be set when data.root is not defined")
        train_paths = read_split_file(train_source)
        val_paths = read_split_file(val_source)

    cache_images = data_cfg.get("cache_images", True)
    val_cache_images = data_cfg.get("val_cache_images", cache_images)

    train_dataset = PixelArtDataset(
        paths=train_paths,
        scale=model_cfg["scale"],
        patch_size_hr=data_cfg["patch_size_hr"],
        degradation_config=degradation_cfg,
        crops_tile_aligned_prob=data_cfg["crops_tile_aligned_prob"],
        flips=data_cfg["flips"],
        rotations=data_cfg["rotations"],
        augment=True,
        cache_images=cache_images,
        patch_size_ref=patch_ref,
    )
    val_dataset = PixelArtDataset(
        paths=val_paths,
        scale=model_cfg["scale"],
        patch_size_hr=data_cfg.get("val_patch_size_hr"),
        degradation_config=degradation_cfg,
        crops_tile_aligned_prob=1.0,
        flips=False,
        rotations=False,
        augment=False,
        cache_images=val_cache_images,
    )

    pin_memory = device.type == "cuda"

    loss_fn = build_loss_fn(config, device)

    train_workers = data_cfg.get("train_workers", 4)
    val_workers = data_cfg.get("val_workers", 2)

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=worker_seed_init,
    )
    if train_workers > 0:
        train_loader_kwargs["prefetch_factor"] = data_cfg.get("prefetch_factor", 2)
        train_loader_kwargs["persistent_workers"] = data_cfg.get("persistent_workers", True)

    val_loader_kwargs = dict(
        dataset=val_dataset,
        batch_size=data_cfg.get("val_batch_size", 1),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_seed_init,
    )
    if val_workers > 0:
        val_prefetch = data_cfg.get("val_prefetch_factor")
        if val_prefetch is None:
            val_prefetch = max(1, data_cfg.get("prefetch_factor", 2) // 2)
        val_loader_kwargs["prefetch_factor"] = val_prefetch
        val_loader_kwargs["persistent_workers"] = data_cfg.get("val_persistent_workers", False)

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    resume_path = args.resume_from
    if resume_path and Path(resume_path).is_file():
        # Safe load and adapt keys from compiled/wrapped checkpoints
        resume_state = _safe_torch_load(resume_path, weights_only=True)
        raw_state = resume_state.get("model", resume_state)
        model_state = model.state_dict()
        adapted, _ = _extract_adapted_weights(raw_state, model_state, allow_tail=True)

        if not adapted:
            raise RuntimeError(f"no compatible tensors when resuming from {resume_path}")
        model_state.update(adapted)
        model.load_state_dict(model_state, strict=True)
        start_step = int(resume_state.get("step", 0))

    if patch_ref is not None and patch_schedule:
        patch_cursor = _patch_index_for_step(start_step, patch_schedule)
        patch_ref.value = patch_schedule[patch_cursor][1]

    # Decide whether to inherit best_psnr from the checkpoint
    best_psnr = None
    epochs_since_improve = 0
    if resume_state and "best_psnr" in resume_state:
        resume_cfg = resume_state.get("config", {})
        same_scale = resume_cfg.get("model", {}).get("scale") == config["model"]["scale"]
        same_crop  = (resume_cfg.get("eval", {}).get("crop_border", 0)
                    == config.get("eval", {}).get("crop_border", 0))
        same_ychan = bool(resume_cfg.get("eval", {}).get("y_channel", False)) \
                    == bool(config.get("eval", {}).get("y_channel", False))
        # Optional: also require same output_dir to be extra safe
        same_out = resume_cfg.get("output_dir") == config.get("output_dir")

        if same_scale and same_crop and same_ychan and same_out:
            best_psnr = float(resume_state["best_psnr"])
            epochs_since_improve = int(resume_state.get("epochs_since_improve", 0))
            print(f"[resume] inheriting best_psnr={best_psnr:.4f}")
        else:
            print("[resume] resetting best_psnr for new run/config (scale/eval/output changed).")
    else:
        pretrained_path = config.get("pretrained_path")
        if pretrained_path and Path(pretrained_path).is_file():
            loaded = _safe_torch_load(pretrained_path, weights_only=True)
            state_dict = loaded["model"] if "model" in loaded else loaded
            model_state = model.state_dict()
            adapted, matched_tail = _extract_adapted_weights(
                state_dict, model_state, allow_tail=True
            )
            if adapted:
                model_state.update(adapted)
                model.load_state_dict(model_state, strict=True)
            tail_status = "tail included" if matched_tail else "tail skipped"
            print(f"warm start matched {len(adapted)} tensors ({tail_status})")
        else:
            print("training from scratch")

    baseline_path = config.get("baseline_path")
    if baseline_path and Path(baseline_path).is_file():
        loaded = _safe_torch_load(baseline_path, weights_only=True)
        baseline_state = loaded["model"] if "model" in loaded else loaded
        baseline_model = EDSR(
            scale=model_cfg["scale"],
            n_feats=model_cfg["n_feats"],
            n_resblocks=model_cfg["n_resblocks"],
            res_scale=model_cfg["res_scale"],
        ).to(device)
        baseline_model_state = baseline_model.state_dict()
        adapted_baseline, _ = _extract_adapted_weights(
            baseline_state, baseline_model_state, allow_tail=True
        )
        if not adapted_baseline:
            print(f"skipped baseline check, no compatible tensors in {baseline_path}")
        else:
            print(f"baseline matched {len(adapted_baseline)} tensors (include tail)")
            adapted_keys = set(adapted_baseline.keys())
            tail_keys = {key for key in baseline_model_state.keys() if key.startswith("tail.")}
            missing_tail = tail_keys - adapted_keys
            if tail_keys and missing_tail:
                loaded_tail = len(tail_keys) - len(missing_tail)
                print(
                    f"skipped baseline check, only loaded {loaded_tail}/{len(tail_keys)} tail "
                    f"tensors from {baseline_path}"
                )
            else:
                baseline_model_state.update(adapted_baseline)
                baseline_model.load_state_dict(baseline_model_state, strict=False)
                b_l1, b_psnr, b_ssim = validate(baseline_model, val_loader, device, config)
                print(
                    f"baseline L1 {b_l1:.6f} | PSNR {b_psnr:.4f} | SSIM {b_ssim:.4f}"
                )


    if config["training"].get("compile", False) and device.type == "cuda":
        model = torch.compile(model)

    opt_cfg = config["optimizer"]

    # Split params
    tail_param_ids = {id(p) for p in model.tail.parameters()}
    trunk_params, tail_params = [], []
    for _, param in model.named_parameters():
        is_tail = id(param) in tail_param_ids
        if tail_only:
            param.requires_grad = is_tail
        if not param.requires_grad:
            continue
        (tail_params if is_tail else trunk_params).append(param)

    if tail_only and not tail_params:
        raise RuntimeError("tail_only is true but no tail parameters were found")

    # Build param groups with names so we can map back to target LR
    param_groups = []
    if not tail_only and trunk_params:
        param_groups.append({"params": trunk_params, "lr": opt_cfg["lr_trunk"], "name": "trunk"})
    if tail_params:
        param_groups.append({"params": tail_params, "lr": opt_cfg["lr_tail"], "name": "tail"})
    if not param_groups:
        raise RuntimeError("No trainable parameters")

    # Optimizer
    opt_type = opt_cfg.get("type", "adamw").lower()
    if opt_type == "adam":
        optimizer = torch.optim.Adam(param_groups, weight_decay=opt_cfg.get("weight_decay", 0.0))
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=opt_cfg.get("weight_decay", 0.0))

    # Target learning rates used for warmup (don’t depend on resume-time values)
    name_to_target = {
        "trunk": opt_cfg.get("lr_trunk", opt_cfg["lr_tail"]),
        "tail":  opt_cfg["lr_tail"],
    }
    target_lrs = [name_to_target.get(g.get("name", "tail"), g["lr"]) for g in optimizer.param_groups]    

    amp_enabled = training_cfg["amp"] and device.type == "cuda"
    if amp_enabled:
        # New API only (PyTorch >= 2.x)
        scaler = torch.amp.GradScaler("cuda")  # positional "cuda", not device_type=...
        autocast_context = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        scaler = None
        autocast_context = nullcontext

    # If resuming, load optimizer (and scaler) state before computing base_lrs
    if resume_state is not None:
        if "optimizer" in resume_state:
            try:
                optimizer.load_state_dict(resume_state["optimizer"])
            except ValueError as e:
                print(f"warning: could not load optimizer state: {e}")
            else:
                for state in optimizer.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device, non_blocking=True)
        if amp_enabled and "scaler" in resume_state and resume_state["scaler"] is not None:
            try:
                scaler.load_state_dict(resume_state["scaler"])
            except Exception as e:
                print(f"warning: could not load AMP scaler state: {e}")

    total_steps = training_cfg["steps"]
    pilot_steps = training_cfg.get("pilot_steps")
    if pilot_mode:
        if pilot_steps is None:
            pilot_steps = training_cfg.get("val_every", total_steps)
        pilot_steps = max(int(pilot_steps), 1)
        total_steps = min(total_steps, pilot_steps)
        print(f"[pilot] limiting training to {total_steps} steps (pilot_steps={pilot_steps})")
    warmup_steps = training_cfg.get("warmup_steps", 0)
    warmup_start_factor = training_cfg.get("warmup_start_factor", 1e-3)
    # compute base_lrs after any resume has potentially changed group LRs
    base_lrs = target_lrs
    default_dir = f"runs/edsr_x{model_cfg['scale']}"
    output_dir = Path(config.get("output_dir", default_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pth"

    # Track resume status for LR warmup decisions
    resume_step = start_step
    resuming = resume_state is not None

    try:
        model.train()
        train_iter = iter(train_loader)
        progress = tqdm(range(start_step + 1, total_steps + 1), initial=start_step, total=total_steps)
        for step in progress:
            if patch_ref is not None and patch_schedule:
                while patch_cursor + 1 < len(patch_schedule) and step >= patch_schedule[patch_cursor + 1][0]:
                    patch_cursor += 1
                    new_patch = patch_schedule[patch_cursor][1]
                    patch_ref.value = new_patch
                    print(f"[patch] step {step} patch_size={new_patch}")
            try:
                lr_batch, hr_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                lr_batch, hr_batch = next(train_iter)

            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Apply warmup only if not resuming past warmup. If we resumed during warmup,
            # continue from the current step; otherwise preserve checkpoint LRs.
            in_warmup = warmup_steps > 0 and step <= warmup_steps
            resuming_past_warmup = resuming and (resume_step >= warmup_steps)

            if in_warmup and not resuming_past_warmup:
                progress_frac = step / warmup_steps
                factor = warmup_start_factor + (1.0 - warmup_start_factor) * progress_frac
                for lr, group in zip(base_lrs, optimizer.param_groups):
                    group["lr"] = lr * factor
            elif warmup_steps > 0 and step == warmup_steps + 1 and not resuming_past_warmup:
                # Ensure exact base LR at the first step after warmup (only when not resuming past warmup)
                for lr, group in zip(base_lrs, optimizer.param_groups):
                    group["lr"] = lr

            if amp_enabled:
                with autocast_context():
                    sr_batch = model(lr_batch)
                    loss = loss_fn(sr_batch, hr_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr_batch = model(lr_batch)
                loss = loss_fn(sr_batch, hr_batch)
                loss.backward()
                optimizer.step()

            if step % 50 == 0 or step == 1:
                progress.set_description(f"step {step} loss {loss.item():.4f}")

            if step % training_cfg["val_every"] == 0 or step == total_steps:
                val_l1, val_psnr, val_ssim = validate(model, val_loader, device, config)
                print(f"validation mean L1: {val_l1:.6f} | PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}")

                improved = best_psnr is None or val_psnr > best_psnr + early_stop_min_delta
                if improved:
                    best_psnr = val_psnr
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1

                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                to_save = {
                    "model": base_model.state_dict(),  # save unwrapped module
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_psnr": best_psnr,
                    "epochs_since_improve": epochs_since_improve,
                    "config": config,
                }
                if amp_enabled:
                    to_save["scaler"] = scaler.state_dict()

                # always write rolling last
                last_path = output_dir / "last.pth"
                torch.save(to_save, last_path)

                if improved:
                    torch.save(to_save, best_path)
                    print(f"saved best PSNR: {val_psnr:.4f} → {best_path}")

                if early_stop_patience > 0 and epochs_since_improve >= early_stop_patience:
                    print(
                        f"[early_stop] no PSNR improvement for {epochs_since_improve} validation runs, stopping at step {step}"
                    )
                    break
    finally:
        if patch_manager is not None:
            try:
                patch_manager.shutdown()
            except Exception as exc:
                print(f"warning: could not shutdown patch manager: {exc}")

def _align_spatial_dims(sr_batch, hr_batch):
    # Ensure tensors share the same spatial extent before computing the loss.
    sr_h, sr_w = sr_batch.shape[-2:]
    hr_h, hr_w = hr_batch.shape[-2:]
    target_h = min(sr_h, hr_h)
    target_w = min(sr_w, hr_w)
    if target_h <= 0 or target_w <= 0:
        raise RuntimeError(
            f"non-positive target size when aligning tensors: {(target_h, target_w)}"
        )
    if sr_h != target_h or sr_w != target_w:
        sr_batch = sr_batch[..., :target_h, :target_w]
    if hr_h != target_h or hr_w != target_w:
        hr_batch = hr_batch[..., :target_h, :target_w]
    return sr_batch, hr_batch


def validate(model, loader, device, config):
    model.eval()
    l1_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    ssim_count = 0
    count = 0

    eval_cfg = config.get("eval", {})
    crop = int(eval_cfg.get("crop_border", 0) or 0)
    use_y = bool(eval_cfg.get("y_channel", False))

    with torch.no_grad():
        for lr_batch, hr_batch in loader:
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            sr_batch = model(lr_batch)
            if sr_batch.shape[-2:] != hr_batch.shape[-2:]:
                sr_batch, hr_batch = _align_spatial_dims(sr_batch, hr_batch)

            sr_batch = sr_batch.clamp_(0.0, 1.0)
            hr_eval = hr_batch
            sr_eval = sr_batch

            if crop > 0:
                sr_eval = sr_eval[:, :, crop:-crop, crop:-crop]
                hr_eval = hr_eval[:, :, crop:-crop, crop:-crop]

            sr_c = sr_eval
            hr_c = hr_eval
            if use_y:
                sr_c = _to_y(sr_c)
                hr_c = _to_y(hr_c)
            n = sr_c.size(0)

            # L1 per sample
            l1_batch = F.l1_loss(sr_c, hr_c, reduction="none").flatten(1).mean(dim=1)
            l1_total += l1_batch.sum().item()

            # PSNR per sample
            mse = ((sr_c - hr_c) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
            batch_psnr = 10.0 * torch.log10(1.0 / mse)
            psnr_total += batch_psnr.sum().item()

            # SSIM: guard tiny tiles (default kernel_size=11 needs h,w>=11). Use odd window <= min(h,w), skip if <3.
            h, w = int(sr_c.shape[-2]), int(sr_c.shape[-1])
            win = min(11, h, w)
            if win >= 3:
                if win % 2 == 0:
                    win -= 1
                ssim_batch = structural_similarity_index_measure(sr_c, hr_c, data_range=1.0, kernel_size=win)
                ssim_total += float(ssim_batch) * n
                ssim_count += n
            # else: skip SSIM accumulation for this batch

            count += n

    model.train()
    mean_l1 = l1_total / max(1, count)
    mean_psnr = psnr_total / max(1, count)
    mean_ssim = ssim_total / max(1, ssim_count)
    return mean_l1, mean_psnr, mean_ssim

if __name__ == "__main__":
    main()
