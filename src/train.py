import argparse
import copy
import math
from contextlib import nullcontext
from multiprocessing import Manager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data import PixelArtDataset, DegradationConfig
from src.data.utils import auto_split_paths, read_split_file
from src.models import EDSR
from src.training import (
    EMAHelper,
    LossComputer,
    SWAHelper,
    apply_residual_output,
    build_degradation_schedule,
    build_patch_schedule,
    damp_optimizer_moments,
    extract_adapted_weights,
    pick_device,
    safe_torch_load,
    schedule_index,
    set_worker_seed_base,
    to_y,
    unwrap_model,
    worker_seed_init,
)
from src.utils.seed import set_seed
from torchmetrics.functional.image import structural_similarity_index_measure


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
    total_steps = int(training_cfg["steps"])
    pilot_steps = training_cfg.get("pilot_steps")
    if pilot_mode:
        if pilot_steps is None:
            pilot_steps = training_cfg.get("val_every", total_steps)
        pilot_steps = max(int(pilot_steps), 1)
        total_steps = min(total_steps, pilot_steps)
        print(f"[pilot] limiting training to {total_steps} steps (pilot_steps={pilot_steps})")
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

    base_degrad_cfg = config["degradations"]
    degradation_cfg = DegradationConfig(**base_degrad_cfg)

    data_cfg = config["data"]
    patch_schedule = build_patch_schedule(data_cfg, total_steps)
    degrad_schedule = build_degradation_schedule(base_degrad_cfg, data_cfg.get("degrad_schedule"), total_steps)
    manager = None
    patch_ref = None
    degrad_ref = None
    patch_cursor = 0
    degrad_cursor = 0
    if patch_schedule or degrad_schedule:
        manager = Manager()
    if patch_schedule:
        patch_ref = manager.Value("i", patch_schedule[0][1])
    if degrad_schedule:
        degrad_ref = manager.Value("i", 0)

    train_degrad_configs = [cfg for _, cfg in degrad_schedule] if degrad_schedule else [degradation_cfg]

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

    channels_last = bool(data_cfg.get("channels_last", True))

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
        degradation_configs=train_degrad_configs,
        degradation_phase_ref=degrad_ref,
        edge_sampling=data_cfg.get("edge_sampling"),
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
        degradation_configs=[degradation_cfg],
    )

    pin_memory = device.type == "cuda"

    loss_computer = LossComputer(config, device, total_steps)

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
        resume_state = safe_torch_load(resume_path, weights_only=True)
        raw_state = resume_state.get("model", resume_state)
        model_state = model.state_dict()
        adapted, _ = extract_adapted_weights(raw_state, model_state, allow_tail=True)

        if not adapted:
            raise RuntimeError(f"no compatible tensors when resuming from {resume_path}")
        model_state.update(adapted)
        model.load_state_dict(model_state, strict=True)
        start_step = int(resume_state.get("step", 0))

    if patch_ref is not None and patch_schedule:
        patch_cursor = schedule_index(start_step, patch_schedule)
        patch_ref.value = patch_schedule[patch_cursor][1]
    if degrad_ref is not None and degrad_schedule:
        degrad_cursor = schedule_index(start_step, degrad_schedule)
        degrad_ref.value = degrad_cursor

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
            loaded = safe_torch_load(pretrained_path, weights_only=True)
            state_dict = loaded["model"] if "model" in loaded else loaded
            model_state = model.state_dict()
            adapted, matched_tail = extract_adapted_weights(
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
        loaded = safe_torch_load(baseline_path, weights_only=True)
        baseline_state = loaded["model"] if "model" in loaded else loaded
        baseline_model = EDSR(
            scale=model_cfg["scale"],
            n_feats=model_cfg["n_feats"],
            n_resblocks=model_cfg["n_resblocks"],
            res_scale=model_cfg["res_scale"],
        ).to(device)
        baseline_model_state = baseline_model.state_dict()
        adapted_baseline, _ = extract_adapted_weights(
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


    ema_helper = None
    ema_cfg = training_cfg.get("ema", {})
    if ema_cfg.get("enabled"):
        ema_model = copy.deepcopy(unwrap_model(model)).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        ema_decay = ema_cfg.get("decay", 0.9999)
        ema_update_every = ema_cfg.get("update_every", 1)
        ema_helper = EMAHelper(ema_model, ema_decay, ema_update_every)
        if resume_state and "ema_state" in resume_state:
            ema_helper.load_state_dict(
                {"model": resume_state["ema_state"], "updates": resume_state.get("ema_updates", 0)}
            )

    swa_helper = None
    swa_cfg = training_cfg.get("swa", {})
    swa_start_step = None
    swa_update_every = None
    if swa_cfg.get("enabled") and ema_helper is not None:
        swa_helper = SWAHelper(ema_helper.model)
        swa_start_step = int(total_steps * float(swa_cfg.get("start_frac", 0.8)))
        swa_update_every = max(1, int(swa_cfg.get("update_every", 5000)))
        if resume_state and "swa_state" in resume_state:
            swa_helper.load_state_dict(
                {"model": resume_state["swa_state"], "samples": resume_state.get("swa_samples", 0)}
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

    distill_cfg = config.get("distillation", {})
    distill_enabled = bool(distill_cfg.get("enabled"))
    teacher_model = None
    teacher_scale = None
    teacher_residual = False
    teacher_twopass = bool(distill_cfg.get("teacher_twopass", False))
    distill_sample_frac = float(distill_cfg.get("sample_frac", 1.0))
    distill_sample_frac = max(0.0, min(1.0, distill_sample_frac))
    distill_weight_max = float(distill_cfg.get("weight_max", 0.0))
    distill_weight_final = float(distill_cfg.get("weight_final", distill_weight_max))
    distill_start_frac = float(distill_cfg.get("start_frac", 0.0))
    distill_final_frac = float(distill_cfg.get("final_frac", 0.9))
    distill_start_step = int(total_steps * distill_start_frac)
    distill_final_step = int(total_steps * distill_final_frac)
    distill_final_step = max(distill_final_step, distill_start_step)
    teacher_path = distill_cfg.get("teacher_path")
    if distill_enabled:
        if not teacher_path or not Path(teacher_path).is_file():
            print("warning: distillation enabled but teacher_path missing; disabling distillation")
            distill_enabled = False
        else:
            teacher_state = safe_torch_load(teacher_path, weights_only=True)
            teacher_model_cfg = teacher_state.get("config", {}).get("model", model_cfg)
            teacher_scale = int(teacher_model_cfg.get("scale", model_cfg["scale"]))
            teacher_residual = bool(teacher_model_cfg.get("residual_output", False))
            teacher_model = EDSR(
                scale=teacher_scale,
                n_feats=teacher_model_cfg.get("n_feats", model_cfg["n_feats"]),
                n_resblocks=teacher_model_cfg.get("n_resblocks", model_cfg["n_resblocks"]),
                res_scale=teacher_model_cfg.get("res_scale", model_cfg["res_scale"]),
            ).to(device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            candidate = (
                teacher_state.get("ema_state")
                or teacher_state.get("model")
                or teacher_state.get("state_dict")
                or teacher_state
            )
            teacher_params = teacher_model.state_dict()
            adapted, _ = extract_adapted_weights(candidate, teacher_params, allow_tail=True)
            if not adapted:
                raise RuntimeError(f"could not adapt teacher checkpoint {teacher_path}")
            teacher_params.update(adapted)
            teacher_model.load_state_dict(teacher_params, strict=True)

    def _distill_weight_for_step(step):
        if not distill_enabled or distill_weight_max <= 0.0:
            return 0.0
        if step < distill_start_step:
            return 0.0
        total_range = max(1, total_steps - distill_start_step)
        progress = min(max((step - distill_start_step) / total_range, 0.0), 1.0)
        weight = distill_weight_max * progress
        if distill_weight_final != distill_weight_max and step >= distill_final_step:
            tail_range = max(1, total_steps - distill_final_step)
            tail = min(max((step - distill_final_step) / tail_range, 0.0), 1.0)
            weight = distill_weight_max + (distill_weight_final - distill_weight_max) * tail
        return max(0.0, weight)

    def _teacher_forward(lr_input):
        if not distill_enabled or teacher_model is None:
            return None
        if teacher_twopass:
            mid = teacher_model(lr_input)
            mid = apply_residual_output(mid, lr_input, teacher_scale, teacher_residual)
            mid = mid.clamp(0.0, 1.0)
            sr = teacher_model(mid)
            sr = apply_residual_output(sr, mid, teacher_scale, teacher_residual)
            return sr
        sr = teacher_model(lr_input)
        return apply_residual_output(sr, lr_input, teacher_scale, teacher_residual)

    def _maybe_apply_distillation(step, sr_batch, lr_batch, loss_tensor, loss_terms):
        if not distill_enabled:
            loss_terms["distill"] = 0.0
            return loss_tensor, 0.0
        weight = _distill_weight_for_step(step)
        if weight <= 0.0:
            loss_terms["distill"] = 0.0
            return loss_tensor, weight
        sample = torch.rand((), device=lr_batch.device).item()
        if sample >= distill_sample_frac:
            loss_terms["distill"] = 0.0
            return loss_tensor, weight
        teacher_out = None
        with torch.no_grad():
            ctx = autocast_context if amp_enabled else nullcontext
            with ctx():
                teacher_out = _teacher_forward(lr_batch)
        if teacher_out is None:
            loss_terms["distill"] = 0.0
            return loss_tensor, weight
        teacher_out = teacher_out.clamp(0.0, 1.0)
        distill_loss = F.mse_loss(sr_batch, teacher_out)
        loss_terms["distill"] = float(distill_loss.detach())
        return loss_tensor + weight * distill_loss, weight
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

    warmup_steps = training_cfg.get("warmup_steps", 0)
    warmup_start_factor = training_cfg.get("warmup_start_factor", 1e-3)
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler_type = (scheduler_cfg.get("type") or "").lower()
    min_lr_factor = float(scheduler_cfg.get("min_lr_factor", 1.0))
    final_floor_factor = float(scheduler_cfg.get("final_min_lr_factor", min_lr_factor))
    final_floor_start = float(scheduler_cfg.get("final_phase_start", 0.9))
    patch_mini_warmup_steps = int(training_cfg.get("patch_mini_warmup_steps", 0))
    mini_warmup_start = None
    mini_warmup_end = None
    def _scheduled_lrs_for_step(step):
        if scheduler_type != "cosine" or step <= warmup_steps:
            return list(base_lrs)
        denom = max(1, total_steps - warmup_steps)
        progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        floor = min_lr_factor
        if final_floor_factor != min_lr_factor and progress >= final_floor_start:
            tail = (progress - final_floor_start) / max(1e-8, 1.0 - final_floor_start)
            tail = min(max(tail, 0.0), 1.0)
            floor = min_lr_factor + (final_floor_factor - min_lr_factor) * tail
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base * (floor + (1.0 - floor) * cosine) for base in base_lrs]
    # compute base_lrs after any resume has potentially changed group LRs
    base_lrs = target_lrs
    trainable_params = [p for p in unwrap_model(model).parameters() if p.requires_grad]
    grad_clip_cfg = training_cfg.get("grad_clip", {})
    grad_clip_enabled = bool(grad_clip_cfg.get("enabled"))
    grad_clip_max = float(grad_clip_cfg.get("max_norm", 0.0))
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
        residual_enabled = bool(model_cfg.get("residual_output", False))
        current_patch = patch_schedule[patch_cursor][1] if patch_schedule else data_cfg.get("patch_size_hr")
        if patch_ref is not None and current_patch is not None:
            patch_ref.value = current_patch
        for step in progress:
            if mini_warmup_end is not None and step > mini_warmup_end:
                mini_warmup_start = None
                mini_warmup_end = None
            if patch_ref is not None and patch_schedule:
                while patch_cursor + 1 < len(patch_schedule) and step >= patch_schedule[patch_cursor + 1][0]:
                    patch_cursor += 1
                    current_patch = patch_schedule[patch_cursor][1]
                    patch_ref.value = current_patch
                    print(f"[patch] step {step} patch_size={current_patch}")
                    if patch_mini_warmup_steps > 0:
                        mini_warmup_start = step
                        mini_warmup_end = step + patch_mini_warmup_steps
                    else:
                        mini_warmup_start = None
                        mini_warmup_end = None
                    damp_optimizer_moments(optimizer, 0.5)
            if degrad_ref is not None and degrad_schedule:
                while degrad_cursor + 1 < len(degrad_schedule) and step >= degrad_schedule[degrad_cursor + 1][0]:
                    degrad_cursor += 1
                    degrad_ref.value = degrad_cursor
                    phase = train_degrad_configs[degrad_cursor]
                    print(
                        f"[degrad] step {step} jpeg={phase.jpeg_prob:.2f} noise={phase.gaussian_noise_prob:.3f}"
                    )
            try:
                lr_batch, hr_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                lr_batch, hr_batch = next(train_iter)

            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)
            if channels_last:
                lr_batch = lr_batch.to(memory_format=torch.channels_last)
                hr_batch = hr_batch.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            in_warmup = warmup_steps > 0 and step <= warmup_steps
            resuming_past_warmup = resuming and (resume_step >= warmup_steps)
            skip_lr_update = resuming_past_warmup and step == start_step + 1
            if not skip_lr_update:
                if in_warmup and not resuming_past_warmup:
                    progress_frac = step / warmup_steps
                    factor = warmup_start_factor + (1.0 - warmup_start_factor) * progress_frac
                    for lr, group in zip(base_lrs, optimizer.param_groups):
                        group["lr"] = lr * factor
                else:
                    scheduled = _scheduled_lrs_for_step(step)
                    if mini_warmup_end is not None and step <= mini_warmup_end and patch_mini_warmup_steps > 0:
                        frac = (step - mini_warmup_start) / max(1, patch_mini_warmup_steps)
                        frac = min(max(frac, 0.0), 1.0)
                        factor = warmup_start_factor + (1.0 - warmup_start_factor) * frac
                        scheduled = [lr * factor for lr in scheduled]
                    for lr_value, group in zip(scheduled, optimizer.param_groups):
                        group["lr"] = lr_value

            if amp_enabled:
                with autocast_context():
                    sr_batch = model(lr_batch)
                    sr_batch = apply_residual_output(sr_batch, lr_batch, model_cfg["scale"], residual_enabled)
                    loss_value, loss_terms = loss_computer.compute(sr_batch, hr_batch, step)
                    loss_value, distill_weight = _maybe_apply_distillation(step, sr_batch, lr_batch, loss_value, loss_terms)
                scaler.scale(loss_value).backward()
                if grad_clip_enabled and grad_clip_max > 0.0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(trainable_params, grad_clip_max)
                scaler.step(optimizer)
                scaler.update()
            else:
                sr_batch = model(lr_batch)
                sr_batch = apply_residual_output(sr_batch, lr_batch, model_cfg["scale"], residual_enabled)
                loss_value, loss_terms = loss_computer.compute(sr_batch, hr_batch, step)
                loss_value, distill_weight = _maybe_apply_distillation(step, sr_batch, lr_batch, loss_value, loss_terms)
                loss_value.backward()
                if grad_clip_enabled and grad_clip_max > 0.0:
                    clip_grad_norm_(trainable_params, grad_clip_max)
                optimizer.step()

            if ema_helper is not None:
                ema_helper.update(unwrap_model(model))
            if swa_helper is not None and ema_helper is not None and swa_start_step is not None:
                if step >= swa_start_step and swa_update_every and step % swa_update_every == 0:
                    swa_helper.update(ema_helper.model)

            if step % 50 == 0 or step == 1:
                loss_num = float(loss_value.detach())
                comp_bits = ",".join(f"{k[:4]}={v:.3f}" for k, v in loss_terms.items())
                lr_bits = ",".join(f"{g.get('name', 'pg')}={g['lr']:.2e}" for g in optimizer.param_groups)
                postfix = {
                    "loss": f"{loss_num:.4f}",
                    "lr": lr_bits,
                    "patch": current_patch,
                    "distill": f"{distill_weight:.3f}",
                }
                progress.set_description(f"step {step} {comp_bits}")
                progress.set_postfix(postfix)

            if step % training_cfg["val_every"] == 0 or step == total_steps:
                eval_model = ema_helper.model if ema_helper is not None else model
                val_l1, val_psnr, val_ssim = validate(eval_model, val_loader, device, config)
                if ema_helper is not None:
                    raw_l1, raw_psnr, raw_ssim = validate(model, val_loader, device, config)
                    print(
                        f"validation EMA L1 {val_l1:.6f} | PSNR {val_psnr:.4f} | SSIM {val_ssim:.4f} || "
                        f"raw L1 {raw_l1:.6f} | PSNR {raw_psnr:.4f} | SSIM {raw_ssim:.4f}"
                    )
                else:
                    print(
                        f"validation mean L1: {val_l1:.6f} | PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}"
                    )

                improved = best_psnr is None or val_psnr > best_psnr + early_stop_min_delta
                if improved:
                    best_psnr = val_psnr
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1

                base_model = unwrap_model(model)
                to_save = {
                    "model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_psnr": best_psnr,
                    "epochs_since_improve": epochs_since_improve,
                    "config": config,
                }
                if ema_helper is not None:
                    to_save["ema_state"] = ema_helper.model.state_dict()
                    to_save["ema_updates"] = ema_helper.updates
                if swa_helper is not None and swa_helper.has_samples():
                    to_save["swa_state"] = swa_helper.model.state_dict()
                    to_save["swa_samples"] = swa_helper.samples
                if amp_enabled:
                    to_save["scaler"] = scaler.state_dict()

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
        if manager is not None:
            try:
                manager.shutdown()
            except Exception as exc:
                print(f"warning: could not shutdown shared manager: {exc}")

    if swa_helper is not None and swa_helper.has_samples():
        swa_l1, swa_psnr, swa_ssim = validate(swa_helper.model, val_loader, device, config)
        swa_payload = {
            "model": swa_helper.model.state_dict(),
            "config": config,
            "swa_samples": swa_helper.samples,
            "metrics": {"l1": swa_l1, "psnr": swa_psnr, "ssim": swa_ssim},
        }
        swa_path = output_dir / "final_swa.pth"
        torch.save(swa_payload, swa_path)
        print(f"[swa] L1 {swa_l1:.6f} | PSNR {swa_psnr:.4f} | SSIM {swa_ssim:.4f} → {swa_path}")
        if best_psnr is None or swa_psnr > best_psnr + early_stop_min_delta:
            best_psnr = swa_psnr
            swa_state = swa_helper.model.state_dict()
            best_payload = {
                "model": swa_state,
                "step": total_steps,
                "best_psnr": best_psnr,
                "epochs_since_improve": epochs_since_improve,
                "config": config,
                "swa_state": swa_state,
                "swa_samples": swa_helper.samples,
                "swa_promoted": True,
            }
            if ema_helper is not None:
                best_payload["ema_state"] = ema_helper.model.state_dict()
                best_payload["ema_updates"] = ema_helper.updates
            torch.save(best_payload, best_path)
            print(f"[swa] improved best checkpoint via SWA ({best_psnr:.4f}) → {best_path}")

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
    was_training = model.training
    model.eval()
    l1_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    ssim_count = 0
    count = 0

    eval_cfg = config.get("eval", {})
    crop = int(eval_cfg.get("crop_border", 0) or 0)
    use_y = bool(eval_cfg.get("y_channel", False))

    residual_enabled = bool(config["model"].get("residual_output", False))
    scale = int(config["model"]["scale"])

    with torch.no_grad():
        for lr_batch, hr_batch in loader:
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            sr_batch = model(lr_batch)
            sr_batch = apply_residual_output(sr_batch, lr_batch, scale, residual_enabled)
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
                sr_c = to_y(sr_c)
                hr_c = to_y(hr_c)
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

    if was_training:
        model.train()
    mean_l1 = l1_total / max(1, count)
    mean_psnr = psnr_total / max(1, count)
    mean_ssim = ssim_total / max(1, ssim_count)
    return mean_l1, mean_psnr, mean_ssim

if __name__ == "__main__":
    main()
