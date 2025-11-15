import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.degradations import generate_lr_from_hr_batch
from src.training import apply_residual_output, damp_optimizer_moments, unwrap_model
from src.training.augment import apply_gpu_augmentations
from src.training.distill import DistillationHelper
from src.training.session import TrainConfig, build_training_state, shutdown_manager
from src.training.validation import validate


def main():
    cfg = TrainConfig()
    state = build_training_state(cfg)
    distiller = DistillationHelper(state)
    try:
        train_loop(state, distiller)
    finally:
        shutdown_manager(state)
    finalize_training(state)


def train_loop(state, distiller):
    model = state.model
    optimizer = state.optimizer
    loss_computer = state.loss_computer
    device = state.device
    channels_last = state.channels_last
    train_loader = state.train_loader
    train_iter = iter(train_loader)

    def reset_train_iterator(full_restart=False):
        nonlocal train_loader, train_iter
        if full_restart:
            old_loader = train_loader
            train_loader = DataLoader(**state.train_loader_kwargs)
            state.train_loader = train_loader
            train_iter = iter(train_loader)
            if old_loader is not None:
                old_iter = getattr(old_loader, "_iterator", None)
                if old_iter is not None:
                    old_iter._shutdown_workers()
        else:
            train_iter = iter(train_loader)

    def next_train_batch():
        nonlocal train_iter
        while True:
            try:
                return next(train_iter)
            except StopIteration:
                reset_train_iterator(full_restart=False)
            except RuntimeError as err:
                message = str(err)
                if "stack expects each tensor to be equal size" in message:
                    print("[loader] mixed tensor sizes detected; resetting train iterator")
                    reset_train_iterator(full_restart=True)
                    continue
                raise

    progress = tqdm(
        range(state.start_step + 1, state.total_steps + 1),
        initial=state.start_step,
        total=state.total_steps,
    )
    residual_enabled = bool(state.model_cfg.get("residual_output", False))
    current_patch = (
        state.patch_schedule[state.patch_cursor][1]
        if state.patch_schedule
        else state.data_cfg.get("patch_size_hr")
    )
    if state.patch_ref is not None and current_patch is not None:
        state.patch_ref.value = current_patch

    for step in progress:
        reset_loader = False
        if state.mini_warmup_end is not None and step > state.mini_warmup_end:
            state.mini_warmup_start = None
            state.mini_warmup_end = None
        if state.patch_ref is not None and state.patch_schedule:
            while (
                state.patch_cursor + 1 < len(state.patch_schedule)
                and step >= state.patch_schedule[state.patch_cursor + 1][0]
            ):
                state.patch_cursor += 1
                current_patch = state.patch_schedule[state.patch_cursor][1]
                state.patch_ref.value = current_patch
                print(f"[patch] step {step} patch_size={current_patch}")
                if state.patch_mini_warmup_steps > 0:
                    state.mini_warmup_start = step
                    state.mini_warmup_end = step + state.patch_mini_warmup_steps
                else:
                    state.mini_warmup_start = None
                    state.mini_warmup_end = None
                damp_optimizer_moments(optimizer, 0.5)
                reset_loader = True
        if state.degrad_ref is not None and state.degrad_schedule:
            while (
                state.degrad_cursor + 1 < len(state.degrad_schedule)
                and step >= state.degrad_schedule[state.degrad_cursor + 1][0]
            ):
                state.degrad_cursor += 1
                state.degrad_ref.value = state.degrad_cursor
                phase = state.train_degrad_configs[state.degrad_cursor]
                print(f"[degrad] step {step} jpeg={phase.jpeg_prob:.2f} noise={phase.gaussian_noise_prob:.3f}")
        if reset_loader:
            reset_train_iterator(full_restart=True)

        batch = next_train_batch()
        if state.train_defer_degradation:
            hr_batch, seed_batch = batch
            hr_batch = hr_batch.to(device, non_blocking=True)
            if channels_last:
                hr_batch = hr_batch.to(memory_format=torch.channels_last)
            seed_values = seed_batch.tolist() if hasattr(seed_batch, "tolist") else list(seed_batch)
            hr_batch = apply_gpu_augmentations(
                hr_batch,
                seed_values,
                bool(state.data_cfg.get("flips", True)),
                bool(state.data_cfg.get("rotations", True)),
            )
            active_degrad_cfg = current_degradation_config(state)
            lr_batch = generate_lr_from_hr_batch(
                hr_batch,
                state.model_cfg["scale"],
                active_degrad_cfg,
                seed_values,
            )
            if channels_last:
                lr_batch = lr_batch.to(memory_format=torch.channels_last)
        else:
            lr_batch, hr_batch = batch
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)
            if channels_last:
                lr_batch = lr_batch.to(memory_format=torch.channels_last)
                hr_batch = hr_batch.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        in_warmup = state.warmup_steps > 0 and step <= state.warmup_steps
        resuming_past_warmup = state.resuming and (state.resume_step >= state.warmup_steps)
        skip_lr_update = resuming_past_warmup and step == state.start_step + 1
        if not skip_lr_update:
            if in_warmup and not resuming_past_warmup:
                progress_frac = step / state.warmup_steps if state.warmup_steps else 1.0
                factor = state.warmup_start_factor + (1.0 - state.warmup_start_factor) * progress_frac
                for lr, group in zip(state.base_lrs, optimizer.param_groups):
                    group["lr"] = lr * factor
            else:
                scheduled = scheduled_lrs_for_step(state, step)
                if (
                    state.mini_warmup_end is not None
                    and step <= state.mini_warmup_end
                    and state.patch_mini_warmup_steps > 0
                ):
                    frac = (step - state.mini_warmup_start) / max(1, state.patch_mini_warmup_steps)
                    frac = min(max(frac, 0.0), 1.0)
                    factor = state.warmup_start_factor + (1.0 - state.warmup_start_factor) * frac
                    scheduled = [lr * factor for lr in scheduled]
                for lr_value, group in zip(scheduled, optimizer.param_groups):
                    group["lr"] = lr_value

        if state.amp_enabled:
            with state.autocast_context():
                sr_batch = model(lr_batch)
                sr_batch = apply_residual_output(sr_batch, lr_batch, state.model_cfg["scale"], residual_enabled)
                loss_value, loss_terms = loss_computer.compute(sr_batch, hr_batch, step)
                loss_value, distill_weight = distiller.maybe_apply(
                    step,
                    lr_batch,
                    sr_batch,
                    loss_value,
                    loss_terms,
                    state.amp_enabled,
                    state.autocast_context,
                )
            state.scaler.scale(loss_value).backward()
            if state.grad_clip_enabled and state.grad_clip_max > 0.0:
                state.scaler.unscale_(optimizer)
                clip_grad_norm_(state.trainable_params, state.grad_clip_max)
            state.scaler.step(optimizer)
            state.scaler.update()
        else:
            sr_batch = model(lr_batch)
            sr_batch = apply_residual_output(sr_batch, lr_batch, state.model_cfg["scale"], residual_enabled)
            loss_value, loss_terms = loss_computer.compute(sr_batch, hr_batch, step)
            loss_value, distill_weight = distiller.maybe_apply(
                step,
                lr_batch,
                sr_batch,
                loss_value,
                loss_terms,
                state.amp_enabled,
                state.autocast_context,
            )
            loss_value.backward()
            if state.grad_clip_enabled and state.grad_clip_max > 0.0:
                clip_grad_norm_(state.trainable_params, state.grad_clip_max)
            optimizer.step()

        if state.ema_helper is not None:
            state.ema_helper.update(unwrap_model(model))
        if (
            state.swa_helper is not None
            and state.ema_helper is not None
            and state.swa_start_step is not None
            and state.swa_update_every
            and step >= state.swa_start_step
            and step % state.swa_update_every == 0
        ):
            state.swa_helper.update(state.ema_helper.model)

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

        perform_validation = step % state.training_cfg["val_every"] == 0 or step == state.total_steps
        if perform_validation:
            eval_model = state.ema_helper.model if state.ema_helper is not None else model
            ema_label = "[val][ema]" if state.ema_helper is not None else "[val]"
            val_l1, val_psnr, val_ssim = validate(
                eval_model,
                state.val_loader,
                device,
                state.config,
                progress_label=ema_label,
            )
            if state.ema_helper is None:
                print(f"validation mean L1: {val_l1:.6f} | PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}")
            else:
                final_eval = step == state.total_steps
                if not final_eval:
                    print(f"validation EMA L1 {val_l1:.6f} | PSNR {val_psnr:.4f} | SSIM {val_ssim:.4f}")
                else:
                    raw_l1, raw_psnr, raw_ssim = validate(
                        model,
                        state.val_loader,
                        device,
                        state.config,
                        progress_label="[val][raw]",
                    )
                    print(
                        f"validation EMA L1 {val_l1:.6f} | PSNR {val_psnr:.4f} | SSIM {val_ssim:.4f} || "
                        f"raw L1 {raw_l1:.6f} | PSNR {raw_psnr:.4f} | SSIM {raw_ssim:.4f}"
                    )

            improved = state.best_psnr is None or val_psnr > state.best_psnr + state.early_stop_min_delta
            if improved:
                state.best_psnr = val_psnr
                state.epochs_since_improve = 0
            else:
                state.epochs_since_improve += 1

            base_model = unwrap_model(model)
            to_save = {
                "model": base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_psnr": state.best_psnr,
                "epochs_since_improve": state.epochs_since_improve,
                "config": state.config,
            }
            if state.ema_helper is not None:
                to_save["ema_state"] = state.ema_helper.model.state_dict()
                to_save["ema_updates"] = state.ema_helper.updates
            if state.swa_helper is not None and state.swa_helper.has_samples():
                to_save["swa_state"] = state.swa_helper.model.state_dict()
                to_save["swa_samples"] = state.swa_helper.samples
            if state.amp_enabled and state.scaler is not None:
                to_save["scaler"] = state.scaler.state_dict()

            last_path = state.output_dir / "last.pth"
            torch.save(to_save, last_path)

            if improved:
                torch.save(to_save, state.best_path)
                print(f"saved best PSNR: {val_psnr:.4f} → {state.best_path}")

            if state.early_stop_patience > 0 and state.epochs_since_improve >= state.early_stop_patience:
                print(
                    f"[early_stop] no PSNR improvement for {state.epochs_since_improve} validation runs, "
                    f"stopping at step {step}"
                )
                break


def finalize_training(state):
    if state.swa_helper is not None and state.swa_helper.has_samples():
        swa_l1, swa_psnr, swa_ssim = validate(state.swa_helper.model, state.val_loader, state.device, state.config)
        swa_payload = {
            "model": state.swa_helper.model.state_dict(),
            "config": state.config,
            "swa_samples": state.swa_helper.samples,
            "metrics": {"l1": swa_l1, "psnr": swa_psnr, "ssim": swa_ssim},
        }
        swa_path = state.output_dir / "final_swa.pth"
        torch.save(swa_payload, swa_path)
        print(f"[swa] L1 {swa_l1:.6f} | PSNR {swa_psnr:.4f} | SSIM {swa_ssim:.4f} → {swa_path}")
        if state.best_psnr is None or swa_psnr > state.best_psnr + state.early_stop_min_delta:
            state.best_psnr = swa_psnr
            swa_state = state.swa_helper.model.state_dict()
            best_payload = {
                "model": swa_state,
                "step": state.total_steps,
                "best_psnr": state.best_psnr,
                "epochs_since_improve": state.epochs_since_improve,
                "config": state.config,
                "swa_state": swa_state,
                "swa_samples": state.swa_helper.samples,
                "swa_promoted": True,
            }
            if state.ema_helper is not None:
                best_payload["ema_state"] = state.ema_helper.model.state_dict()
                best_payload["ema_updates"] = state.ema_helper.updates
            torch.save(best_payload, state.best_path)
            print(f"[swa] improved best checkpoint via SWA ({state.best_psnr:.4f}) → {state.best_path}")


def current_degradation_config(state):
    configs = state.train_degrad_configs
    if not configs:
        return state.degradation_cfg
    if state.degrad_ref is not None:
        idx = int(state.degrad_ref.value)
    else:
        idx = state.degrad_cursor
    idx = max(0, min(idx, len(configs) - 1))
    return configs[idx]


def scheduled_lrs_for_step(state, step):
    if state.scheduler_type != "cosine" or step <= state.warmup_steps:
        return list(state.base_lrs)
    denom = max(1, state.total_steps - state.warmup_steps)
    progress = min(max((step - state.warmup_steps) / denom, 0.0), 1.0)
    floor = state.min_lr_factor
    if state.final_floor_factor != state.min_lr_factor and progress >= state.final_floor_start:
        tail = (progress - state.final_floor_start) / max(1e-8, 1.0 - state.final_floor_start)
        tail = min(max(tail, 0.0), 1.0)
        floor = state.min_lr_factor + (state.final_floor_factor - state.min_lr_factor) * tail
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return [base * (floor + (1.0 - floor) * cosine) for base in state.base_lrs]


if __name__ == "__main__":
    main()
