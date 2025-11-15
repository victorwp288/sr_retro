import argparse
import copy
from contextlib import nullcontext
from multiprocessing import Manager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
import yaml

from src.data import PixelArtDataset, DegradationConfig
from src.data.utils import auto_split_paths, read_split_file
from src.models import EDSR
from src.training import (
    EMAHelper,
    LossComputer,
    SWAHelper,
    build_degradation_schedule,
    build_patch_schedule,
    extract_adapted_weights,
    pick_device,
    safe_torch_load,
    schedule_index,
    set_worker_seed_base,
    unwrap_model,
    worker_seed_init,
)
from src.training.validation import validate
from src.utils.seed import set_seed


class TrainConfig:
    __slots__ = (
        "config_path",
        "config",
        "resume_path",
        "pilot_mode",
        "pilot_steps",
        "total_steps",
        "tail_only",
        "early_stop_patience",
        "early_stop_min_delta",
        "seed",
        "training",
        "model_cfg",
        "data_cfg",
        "base_degrad_cfg",
    )

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--resume_from", default=None)
        parser.add_argument(
            "--pilot",
            action="store_true",
            help="Enable pilot mode by capping training steps for quick sweeps.",
        )
        args = parser.parse_args()

        self.config_path = Path(args.config)
        self.resume_path = args.resume_from
        self.pilot_mode = bool(args.pilot)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.training = self.config["training"]
        self.model_cfg = self.config["model"]
        self.data_cfg = self.config["data"]
        self.base_degrad_cfg = self.config["degradations"]
        self.seed = self.config.get("seed", 42)
        self.tail_only = bool(self.training.get("tail_only", True))
        self.early_stop_patience = int(self.training.get("early_stop_patience", 0) or 0)
        self.early_stop_min_delta = float(self.training.get("early_stop_min_delta", 0.0))

        total_steps = int(self.training["steps"])
        pilot_steps = self.training.get("pilot_steps")
        if self.pilot_mode:
            if pilot_steps is None:
                pilot_steps = self.training.get("val_every", total_steps)
            pilot_steps = max(int(pilot_steps), 1)
            total_steps = min(total_steps, pilot_steps)
        self.total_steps = total_steps
        self.pilot_steps = pilot_steps


def build_training_state(cfg):
    state = SimpleNamespace()
    state.cfg = cfg
    state.config = cfg.config
    state.training_cfg = cfg.training
    state.model_cfg = cfg.model_cfg
    state.data_cfg = cfg.data_cfg
    state.base_degrad_cfg = cfg.base_degrad_cfg
    state.total_steps = cfg.total_steps
    state.tail_only = cfg.tail_only
    state.early_stop_patience = cfg.early_stop_patience
    state.early_stop_min_delta = cfg.early_stop_min_delta
    state.resume_path = cfg.resume_path
    state.resume_state = None
    state.start_step = 0
    state.best_psnr = None
    state.epochs_since_improve = 0
    if cfg.pilot_mode:
        pilot_display = cfg.pilot_steps or state.training_cfg.get("val_every", state.total_steps)
        print(f"[pilot] limiting training to {state.total_steps} steps (pilot_steps={pilot_display})")
    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    set_worker_seed_base(cfg.seed)

    state.device = pick_device()
    print(f"training on {state.device}")
    state.model = EDSR(
        scale=state.model_cfg["scale"],
        n_feats=state.model_cfg["n_feats"],
        n_resblocks=state.model_cfg["n_resblocks"],
        res_scale=state.model_cfg["res_scale"],
    ).to(state.device)

    state.degradation_cfg = DegradationConfig(**state.base_degrad_cfg)

    state.patch_schedule = build_patch_schedule(state.data_cfg, state.total_steps)
    state.degrad_schedule = build_degradation_schedule(
        state.base_degrad_cfg, state.data_cfg.get("degrad_schedule"), state.total_steps
    )
    state.manager = None
    state.patch_ref = None
    state.degrad_ref = None
    state.patch_cursor = 0
    state.degrad_cursor = 0
    if state.patch_schedule or state.degrad_schedule:
        state.manager = Manager()
    if state.patch_schedule:
        state.patch_ref = state.manager.Value("i", state.patch_schedule[0][1])
    if state.degrad_schedule:
        state.degrad_ref = state.manager.Value("i", 0)

    state.train_degrad_configs = (
        [entry for _, entry in state.degrad_schedule] if state.degrad_schedule else [state.degradation_cfg]
    )

    auto_root = state.data_cfg.get("root")
    ratios = state.data_cfg.get("auto_split_ratio", {"train": 0.8, "val": 0.1, "test": 0.1})
    split_seed = state.data_cfg.get("auto_split_seed", cfg.seed)
    if auto_root:
        splits = auto_split_paths(auto_root, ratios, split_seed)
        state.train_paths = splits["train"]
        state.val_paths = splits["val"]
    else:
        train_source = state.data_cfg.get("train_split")
        val_source = state.data_cfg.get("val_split")
        if not train_source or not val_source:
            raise RuntimeError("train_split and val_split must be set when data.root is not defined")
        state.train_paths = read_split_file(train_source)
        state.val_paths = read_split_file(val_source)

    cache_images = state.data_cfg.get("cache_images", True)
    val_cache_images = state.data_cfg.get("val_cache_images", cache_images)

    state.channels_last = bool(state.data_cfg.get("channels_last", True))
    state.train_defer_degradation = bool(state.data_cfg.get("defer_degradation", False))

    edge_meta_root = state.data_cfg.get("edge_metadata_root")
    edge_meta_source = state.data_cfg.get("edge_metadata_source_root")

    state.train_dataset = PixelArtDataset(
        paths=state.train_paths,
        scale=state.model_cfg["scale"],
        patch_size_hr=state.data_cfg["patch_size_hr"],
        degradation_config=state.degradation_cfg,
        crops_tile_aligned_prob=state.data_cfg["crops_tile_aligned_prob"],
        flips=state.data_cfg["flips"],
        rotations=state.data_cfg["rotations"],
        augment=not state.train_defer_degradation,
        cache_images=cache_images,
        patch_size_ref=state.patch_ref,
        degradation_configs=state.train_degrad_configs,
        degradation_phase_ref=state.degrad_ref,
        edge_sampling=state.data_cfg.get("edge_sampling"),
        defer_degradation=state.train_defer_degradation,
        edge_metadata_root=edge_meta_root,
        edge_metadata_source_root=edge_meta_source,
    )
    state.val_dataset = PixelArtDataset(
        paths=state.val_paths,
        scale=state.model_cfg["scale"],
        patch_size_hr=state.data_cfg.get("val_patch_size_hr"),
        degradation_config=state.degradation_cfg,
        crops_tile_aligned_prob=1.0,
        flips=False,
        rotations=False,
        augment=False,
        cache_images=val_cache_images,
        degradation_configs=[state.degradation_cfg],
        defer_degradation=False,
        edge_metadata_root=edge_meta_root,
        edge_metadata_source_root=edge_meta_source,
    )

    state.pin_memory = state.device.type == "cuda"
    state.loss_computer = LossComputer(state.config, state.device, state.total_steps)

    train_workers = state.data_cfg.get("train_workers", 4)
    val_workers = state.data_cfg.get("val_workers", 2)

    state.train_loader_kwargs = dict(
        dataset=state.train_dataset,
        batch_size=state.training_cfg["batch_size"],
        shuffle=True,
        num_workers=train_workers,
        pin_memory=state.pin_memory,
        drop_last=True,
        worker_init_fn=worker_seed_init,
    )
    if train_workers > 0:
        state.train_loader_kwargs["prefetch_factor"] = state.data_cfg.get("prefetch_factor", 2)
        state.train_loader_kwargs["persistent_workers"] = state.data_cfg.get("persistent_workers", True)

    state.val_loader_kwargs = dict(
        dataset=state.val_dataset,
        batch_size=state.data_cfg.get("val_batch_size", 1),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=state.pin_memory,
        drop_last=False,
        worker_init_fn=worker_seed_init,
    )
    if val_workers > 0:
        val_prefetch = state.data_cfg.get("val_prefetch_factor")
        if val_prefetch is None:
            val_prefetch = max(1, state.data_cfg.get("prefetch_factor", 2) // 2)
        state.val_loader_kwargs["prefetch_factor"] = val_prefetch
        state.val_loader_kwargs["persistent_workers"] = state.data_cfg.get("val_persistent_workers", False)

    state.train_loader = DataLoader(**state.train_loader_kwargs)
    state.val_loader = DataLoader(**state.val_loader_kwargs)

    if state.resume_path and Path(state.resume_path).is_file():
        state.resume_state = safe_torch_load(state.resume_path, weights_only=True)
        raw_state = state.resume_state.get("model", state.resume_state)
        model_state = state.model.state_dict()
        adapted, _ = extract_adapted_weights(raw_state, model_state, allow_tail=True)
        if not adapted:
            raise RuntimeError(f"no compatible tensors when resuming from {state.resume_path}")
        model_state.update(adapted)
        state.model.load_state_dict(model_state, strict=True)
        state.start_step = int(state.resume_state.get("step", 0))

    if state.patch_ref is not None and state.patch_schedule:
        state.patch_cursor = schedule_index(state.start_step, state.patch_schedule)
        state.patch_ref.value = state.patch_schedule[state.patch_cursor][1]
    if state.degrad_ref is not None and state.degrad_schedule:
        state.degrad_cursor = schedule_index(state.start_step, state.degrad_schedule)
        state.degrad_ref.value = state.degrad_cursor

    state.best_psnr = None
    state.epochs_since_improve = 0
    if state.resume_state and "best_psnr" in state.resume_state:
        resume_cfg = state.resume_state.get("config", {})
        same_scale = resume_cfg.get("model", {}).get("scale") == state.model_cfg["scale"]
        same_crop = (resume_cfg.get("eval", {}).get("crop_border", 0)
                    == state.config.get("eval", {}).get("crop_border", 0))
        same_ychan = bool(resume_cfg.get("eval", {}).get("y_channel", False)) \
                    == bool(state.config.get("eval", {}).get("y_channel", False))
        same_out = resume_cfg.get("output_dir") == state.config.get("output_dir")
        if same_scale and same_crop and same_ychan and same_out:
            state.best_psnr = float(state.resume_state["best_psnr"])
            state.epochs_since_improve = int(state.resume_state.get("epochs_since_improve", 0))
            print(f"[resume] inheriting best_psnr={state.best_psnr:.4f}")
        else:
            print("[resume] resetting best_psnr for new run/config (scale/eval/output changed).")
    else:
        pretrained_path = state.config.get("pretrained_path")
        if pretrained_path and Path(pretrained_path).is_file():
            loaded = safe_torch_load(pretrained_path, weights_only=True)
            state_dict = loaded["model"] if "model" in loaded else loaded
            model_state = state.model.state_dict()
            adapted, matched_tail = extract_adapted_weights(
                state_dict, model_state, allow_tail=True
            )
            if adapted:
                model_state.update(adapted)
                state.model.load_state_dict(model_state, strict=True)
            tail_status = "tail included" if matched_tail else "tail skipped"
            print(f"warm start matched {len(adapted)} tensors ({tail_status})")
        else:
            print("training from scratch")

    baseline_path = state.config.get("baseline_path")
    if baseline_path and Path(baseline_path).is_file():
        loaded = safe_torch_load(baseline_path, weights_only=True)
        baseline_state = loaded["model"] if "model" in loaded else loaded
        baseline_model = EDSR(
            scale=state.model_cfg["scale"],
            n_feats=state.model_cfg["n_feats"],
            n_resblocks=state.model_cfg["n_resblocks"],
            res_scale=state.model_cfg["res_scale"],
        ).to(state.device)
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
                b_l1, b_psnr, b_ssim = validate(baseline_model, state.val_loader, state.device, state.config)
                print(
                    f"baseline L1 {b_l1:.6f} | PSNR {b_psnr:.4f} | SSIM {b_ssim:.4f}"
                )

    state.ema_helper = None
    ema_cfg = state.training_cfg.get("ema", {})
    if ema_cfg.get("enabled"):
        ema_model = copy.deepcopy(unwrap_model(state.model)).to(state.device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        ema_decay = ema_cfg.get("decay", 0.9999)
        ema_update_every = ema_cfg.get("update_every", 1)
        state.ema_helper = EMAHelper(ema_model, ema_decay, ema_update_every)
        if state.resume_state and "ema_state" in state.resume_state:
            state.ema_helper.load_state_dict(
                {"model": state.resume_state["ema_state"], "updates": state.resume_state.get("ema_updates", 0)}
            )

    state.swa_helper = None
    swa_cfg = state.training_cfg.get("swa", {})
    state.swa_start_step = None
    state.swa_update_every = None
    if swa_cfg.get("enabled") and state.ema_helper is not None:
        state.swa_helper = SWAHelper(state.ema_helper.model)
        state.swa_start_step = int(state.total_steps * float(swa_cfg.get("start_frac", 0.8)))
        state.swa_update_every = max(1, int(swa_cfg.get("update_every", 5000)))
        if state.resume_state and "swa_state" in state.resume_state:
            state.swa_helper.load_state_dict(
                {"model": state.resume_state["swa_state"], "samples": state.resume_state.get("swa_samples", 0)}
            )

    if state.training_cfg.get("compile", False) and state.device.type == "cuda":
        state.model = torch.compile(state.model)

    opt_cfg = state.config["optimizer"]
    tail_param_ids = {id(p) for p in state.model.tail.parameters()}
    trunk_params, tail_params = [], []
    for _, param in state.model.named_parameters():
        is_tail = id(param) in tail_param_ids
        if state.tail_only:
            param.requires_grad = is_tail
        if not param.requires_grad:
            continue
        (tail_params if is_tail else trunk_params).append(param)

    if state.tail_only and not tail_params:
        raise RuntimeError("tail_only is true but no tail parameters were found")

    param_groups = []
    if not state.tail_only and trunk_params:
        param_groups.append({"params": trunk_params, "lr": opt_cfg["lr_trunk"], "name": "trunk"})
    if tail_params:
        param_groups.append({"params": tail_params, "lr": opt_cfg["lr_tail"], "name": "tail"})
    if not param_groups:
        raise RuntimeError("No trainable parameters")

    opt_type = opt_cfg.get("type", "adamw").lower()
    if opt_type == "adam":
        state.optimizer = torch.optim.Adam(param_groups, weight_decay=opt_cfg.get("weight_decay", 0.0))
    else:
        state.optimizer = torch.optim.AdamW(param_groups, weight_decay=opt_cfg.get("weight_decay", 0.0))

    state.name_to_target = {
        "trunk": opt_cfg.get("lr_trunk", opt_cfg["lr_tail"]),
        "tail": opt_cfg["lr_tail"],
    }
    state.target_lrs = [state.name_to_target.get(g.get("name", "tail"), g["lr"]) for g in state.optimizer.param_groups]

    state.amp_enabled = state.training_cfg["amp"] and state.device.type == "cuda"
    if state.amp_enabled:
        state.scaler = torch.amp.GradScaler("cuda")
        state.autocast_context = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        state.scaler = None
        state.autocast_context = nullcontext

    state.distill_cfg = state.config.get("distillation", {})
    state.distill_enabled = bool(state.distill_cfg.get("enabled"))
    state.teacher_model = None
    state.teacher_scale = None
    state.teacher_residual = False
    state.teacher_twopass = bool(state.distill_cfg.get("teacher_twopass", False))
    state.distill_sample_frac = float(state.distill_cfg.get("sample_frac", 1.0))
    state.distill_sample_frac = max(0.0, min(1.0, state.distill_sample_frac))
    state.distill_weight_max = float(state.distill_cfg.get("weight_max", 0.0))
    state.distill_weight_final = float(state.distill_cfg.get("weight_final", state.distill_weight_max))
    distill_start_frac = float(state.distill_cfg.get("start_frac", 0.0))
    distill_final_frac = float(state.distill_cfg.get("final_frac", 0.9))
    state.distill_start_step = int(state.total_steps * distill_start_frac)
    state.distill_final_step = int(state.total_steps * distill_final_frac)
    state.distill_final_step = max(state.distill_final_step, state.distill_start_step)
    teacher_path = state.distill_cfg.get("teacher_path")
    if state.distill_enabled:
        if not teacher_path or not Path(teacher_path).is_file():
            print("warning: distillation enabled but teacher_path missing; disabling distillation")
            state.distill_enabled = False
        else:
            teacher_state = safe_torch_load(teacher_path, weights_only=True)
            teacher_model_cfg = teacher_state.get("config", {}).get("model", state.model_cfg)
            state.teacher_scale = int(teacher_model_cfg.get("scale", state.model_cfg["scale"]))
            state.teacher_residual = bool(teacher_model_cfg.get("residual_output", False))
            state.teacher_model = EDSR(
                scale=state.teacher_scale,
                n_feats=teacher_model_cfg.get("n_feats", state.model_cfg["n_feats"]),
                n_resblocks=teacher_model_cfg.get("n_resblocks", state.model_cfg["n_resblocks"]),
                res_scale=teacher_model_cfg.get("res_scale", state.model_cfg["res_scale"]),
            ).to(state.device)
            state.teacher_model.eval()
            for param in state.teacher_model.parameters():
                param.requires_grad = False
            candidate = (
                teacher_state.get("ema_state")
                or teacher_state.get("model")
                or teacher_state.get("state_dict")
                or teacher_state
            )
            teacher_params = state.teacher_model.state_dict()
            adapted, _ = extract_adapted_weights(candidate, teacher_params, allow_tail=True)
            if not adapted:
                raise RuntimeError(f"could not adapt teacher checkpoint {teacher_path}")
            teacher_params.update(adapted)
            state.teacher_model.load_state_dict(teacher_params, strict=True)

    if state.resume_state is not None:
        if "optimizer" in state.resume_state:
            try:
                state.optimizer.load_state_dict(state.resume_state["optimizer"])
            except ValueError as e:
                print(f"warning: could not load optimizer state: {e}")
            else:
                for opt_state in state.optimizer.state.values():
                    for k, v in list(opt_state.items()):
                        if isinstance(v, torch.Tensor):
                            opt_state[k] = v.to(state.device, non_blocking=True)
        if state.amp_enabled and "scaler" in state.resume_state and state.resume_state["scaler"] is not None:
            try:
                state.scaler.load_state_dict(state.resume_state["scaler"])
            except Exception as e:
                print(f"warning: could not load AMP scaler state: {e}")

    state.warmup_steps = state.training_cfg.get("warmup_steps", 0)
    state.warmup_start_factor = state.training_cfg.get("warmup_start_factor", 1e-3)
    scheduler_cfg = state.training_cfg.get("scheduler", {})
    state.scheduler_type = (scheduler_cfg.get("type") or "").lower()
    state.min_lr_factor = float(scheduler_cfg.get("min_lr_factor", 1.0))
    state.final_floor_factor = float(scheduler_cfg.get("final_min_lr_factor", state.min_lr_factor))
    state.final_floor_start = float(scheduler_cfg.get("final_phase_start", 0.9))
    state.patch_mini_warmup_steps = int(state.training_cfg.get("patch_mini_warmup_steps", 0))
    state.mini_warmup_start = None
    state.mini_warmup_end = None
    state.base_lrs = state.target_lrs
    state.trainable_params = [p for p in unwrap_model(state.model).parameters() if p.requires_grad]
    grad_clip_cfg = state.training_cfg.get("grad_clip", {})
    state.grad_clip_enabled = bool(grad_clip_cfg.get("enabled"))
    state.grad_clip_max = float(grad_clip_cfg.get("max_norm", 0.0))
    default_dir = f"runs/edsr_x{state.model_cfg['scale']}"
    state.output_dir = Path(state.config.get("output_dir", default_dir))
    state.output_dir.mkdir(parents=True, exist_ok=True)
    state.best_path = state.output_dir / "best.pth"
    state.resume_step = state.start_step
    state.resuming = state.resume_state is not None
    return state


def shutdown_manager(state):
    if state.manager is not None:
        try:
            state.manager.shutdown()
        except Exception as exc:
            print(f"warning: could not shutdown shared manager: {exc}")
