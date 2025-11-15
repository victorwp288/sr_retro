import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure

from src.training import apply_residual_output, to_y


def _align_spatial_dims(sr_batch, hr_batch):
    sr_h, sr_w = sr_batch.shape[-2:]
    hr_h, hr_w = hr_batch.shape[-2:]
    target_h = min(sr_h, hr_h)
    target_w = min(sr_w, hr_w)
    if target_h <= 0 or target_w <= 0:
        raise RuntimeError(f"non-positive target size when aligning tensors: {(target_h, target_w)}")
    if sr_h != target_h or sr_w != target_w:
        sr_batch = sr_batch[..., :target_h, :target_w]
    if hr_h != target_h or hr_w != target_w:
        hr_batch = hr_batch[..., :target_h, :target_w]
    return sr_batch, hr_batch


def validate(model, loader, device, config, progress_label="[val]"):
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

    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = 0
    report_stride = max(1, total_batches // 5) if total_batches else 1
    next_report = report_stride

    with torch.no_grad():
        for batch_idx, (lr_batch, hr_batch) in enumerate(loader, 1):
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

            l1_batch = F.l1_loss(sr_c, hr_c, reduction="none").flatten(1).mean(dim=1)
            l1_total += l1_batch.sum().item()

            mse = ((sr_c - hr_c) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
            batch_psnr = 10.0 * torch.log10(1.0 / mse)
            psnr_total += batch_psnr.sum().item()

            h, w = int(sr_c.shape[-2]), int(sr_c.shape[-1])
            win = min(11, h, w)
            if win >= 3:
                if win % 2 == 0:
                    win -= 1
                ssim_batch = structural_similarity_index_measure(sr_c, hr_c, data_range=1.0, kernel_size=win)
                ssim_total += float(ssim_batch) * n
                ssim_count += n

            count += n

            if total_batches and (batch_idx >= next_report or batch_idx == total_batches):
                print(f"{progress_label} {batch_idx}/{total_batches}")
                next_report += report_stride

    if was_training:
        model.train()
    mean_l1 = l1_total / max(1, count)
    mean_psnr = psnr_total / max(1, count)
    mean_ssim = ssim_total / max(1, ssim_count)
    return mean_l1, mean_psnr, mean_ssim
