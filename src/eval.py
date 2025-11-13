import argparse
import csv
import inspect
import json
import math
import random
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torchmetrics.image import peak_signal_noise_ratio, structural_similarity_index_measure

from src.data.degradations import DegradationConfig, downscale_with_profile
from src.data.utils import load_image, tensor_from_pil, resolve_paths_from_source
from src.models import EDSR
from src.losses import to_y as to_y_channel


_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters
_SOBEL_KERNEL_X = torch.tensor(
    [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)
_SOBEL_KERNEL_Y = _SOBEL_KERNEL_X.transpose(2, 3).contiguous()


def _safe_torch_load(path):
    kwargs = {"map_location": "cpu"}
    if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def pick_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one or more EDSR checkpoints.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--models", nargs="+", required=True, help="label=ckpt.pth or ckpt.pth entries.")
    parser.add_argument("--out", default="eval_out", help="Output directory.")
    parser.add_argument("--dump", type=int, default=4, help="Number of example strips per model.")
    parser.add_argument("--y", action="store_true", help="Evaluate on Y channel.")
    parser.add_argument("--crop", type=int, default=0, help="Border crop in pixels.")
    parser.add_argument("--device", default=None, help="Device override (cpu, cuda, mps, ...).")
    parser.add_argument("--no_lpips", action="store_true", help="Disable LPIPS metric.")
    parser.add_argument("--lpips_net", default="alex", help="LPIPS backbone (alex, vgg, squeeze).")
    return parser.parse_args()


def parse_model_specs(items):
    specs = []
    for entry in items:
        t = entry.strip()
        if not t:
            continue
        # split into label and rhs first
        if "=" in t:
            label, rhs = t.split("=", 1)
            label = (label or "").strip()
            ckpt = rhs.strip()
        else:
            ckpt = t
            label = Path(ckpt).stem
        prefixes = []
        while True:
            parts = ckpt.split(":", 1)
            if len(parts) == 1:
                break
            prefix, remainder = parts
            if prefix in {"twopass", "ema", "swa", "raw"}:
                prefixes.append(prefix)
                ckpt = remainder
            else:
                break
        mode = "normal"
        variant = "model"
        for prefix in prefixes:
            if prefix == "twopass":
                mode = "twopass"
            elif prefix == "ema":
                variant = "ema"
            elif prefix == "swa":
                variant = "swa"
            elif prefix == "raw":
                variant = "model"
        if not label:
            label = Path(ckpt).stem
        specs.append((label, Path(ckpt), mode, variant))
    return specs

    


def load_config(path):
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text())


def _tensor_mapping(candidate):
    if not hasattr(candidate, "items"):
        return False
    return all(torch.is_tensor(value) for value in candidate.values())


def _select_state_dict(state):
    if isinstance(state, dict):
        for key in ("model", "state_dict", "net", "ema"):
            candidate = state.get(key)
            if _tensor_mapping(candidate):
                return candidate
        if _tensor_mapping(state):
            return state
    return state


def _apply_residual_output(sr, lr, scale, enabled):
    if not enabled or scale <= 1:
        return sr
    target_h = lr.shape[-2] * scale
    target_w = lr.shape[-1] * scale
    base = F.interpolate(lr, size=(target_h, target_w), mode="nearest")
    return sr + base


def _apply_transform(tensor, transpose, flip_h, flip_v):
    out = tensor
    if transpose:
        out = out.transpose(-2, -1)
    if flip_h:
        out = out.flip(-1)
    if flip_v:
        out = out.flip(-2)
    return out


def _inverse_transform(tensor, transpose, flip_h, flip_v):
    out = tensor
    if flip_v:
        out = out.flip(-2)
    if flip_h:
        out = out.flip(-1)
    if transpose:
        out = out.transpose(-2, -1)
    return out


def compute_grad_l1(sr, hr):
    sr_y = to_y_channel(sr).float()
    hr_y = to_y_channel(hr).float()
    kx = _SOBEL_KERNEL_X.to(device=sr_y.device, dtype=torch.float32)
    ky = _SOBEL_KERNEL_Y.to(device=sr_y.device, dtype=torch.float32)
    grad_sr_x = F.conv2d(sr_y, kx, padding=1)
    grad_hr_x = F.conv2d(hr_y, kx, padding=1)
    grad_sr_y = F.conv2d(sr_y, ky, padding=1)
    grad_hr_y = F.conv2d(hr_y, ky, padding=1)
    diff = torch.abs(grad_sr_x - grad_hr_x) + torch.abs(grad_sr_y - grad_hr_y)
    return diff.mean().item()


def run_inference(model, lr, scale, residual_enabled, tta_enabled, mode):
    def single_pass(inp):
        out = model(inp)
        return _apply_residual_output(out, inp, scale, residual_enabled)

    def forward(inp):
        if mode == "twopass":
            mid = single_pass(inp).clamp(0.0, 1.0)
            return single_pass(mid)
        return single_pass(inp)

    sr = forward(lr)
    if not tta_enabled:
        return sr.clamp(0.0, 1.0)

    outputs = []
    for transpose in (False, True):
        for flip_h in (False, True):
            for flip_v in (False, True):
                aug = _apply_transform(lr, transpose, flip_h, flip_v)
                pred = forward(aug)
                pred = _inverse_transform(pred, transpose, flip_h, flip_v)
                outputs.append(pred)
    stacked = torch.stack(outputs, dim=0).mean(dim=0)
    return stacked.clamp(0.0, 1.0)


def resolve_eval_paths(config):
    data_cfg = config["data"]
    source = data_cfg.get("root") or data_cfg.get("val_split") or data_cfg.get("test_split")
    ratios = data_cfg.get("auto_split_ratio")
    seed = data_cfg.get("auto_split_seed", config.get("seed", 42))
    resolved = resolve_paths_from_source(source, subset="test", ratios=ratios, seed=seed)
    return sorted(Path(path) for path in resolved)


def build_model(model_cfg, device):
    model = EDSR(
        scale=model_cfg["scale"],
        n_feats=model_cfg["n_feats"],
        n_resblocks=model_cfg["n_resblocks"],
        res_scale=model_cfg["res_scale"],
    ).to(device)
    model.eval()
    return model


def load_checkpoint(model, ckpt_path, variant, state=None):
    if state is None:
        state = _safe_torch_load(ckpt_path)
    if variant == "ema" and isinstance(state, dict) and "ema_state" in state:
        state_dict = state["ema_state"]
    elif variant == "swa" and isinstance(state, dict) and "swa_state" in state:
        state_dict = state["swa_state"]
    else:
        state_dict = _select_state_dict(state)
        if variant in ("ema", "swa"):
            print(f"warning: checkpoint {ckpt_path} missing {variant} weights, using base model state")
    if hasattr(state_dict, "items"):
        cleaned = {}
        for key, value in state_dict.items():
            name = key
            for prefix in ("module.", "_orig_mod."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            cleaned[name] = value
        state_dict = cleaned
    model.load_state_dict(state_dict, strict=True)
    return state

def _clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        cleaned[k] = v
    return cleaned

def center_crop_to_match(tensor_a, tensor_b):
    target_h = min(tensor_a.shape[-2], tensor_b.shape[-2])
    target_w = min(tensor_a.shape[-1], tensor_b.shape[-1])
    return _center_crop(tensor_a, target_h, target_w), _center_crop(tensor_b, target_h, target_w)


def _center_crop(tensor, target_h, target_w):
    h, w = tensor.shape[-2:]
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return tensor[..., start_h:start_h + target_h, start_w:start_w + target_w]


def crop_border_tensor(tensor, border):
    border = max(0, int(border))
    if border == 0:
        return tensor
    h, w = tensor.shape[-2:]
    limit = min(h, w) // 2
    border = min(border, limit)
    if border == 0:
        return tensor
    return tensor[..., border:h - border, border:w - border]


def compute_psnr(sr, hr):
    return peak_signal_noise_ratio(sr, hr, data_range=1.0).item()


def compute_ssim(sr, hr):
    height, width = sr.shape[-2:]
    kernel = min(11, height, width)
    if kernel < 3:
        return None
    if kernel % 2 == 0:
        kernel -= 1
    if kernel < 3:
        return None
    value = structural_similarity_index_measure(sr, hr, data_range=1.0, kernel_size=kernel)
    return value.item()


def compute_l1(sr, hr):
    return torch.abs(sr - hr).mean().item()


def build_lpips(net, device):
    import lpips

    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def compute_lpips(sr, hr, model):
    if model is None:
        return None
    if sr.size(1) != 3 or hr.size(1) != 3:
        return None
    sr_lp = sr.mul(2.0).sub(1.0)
    hr_lp = hr.mul(2.0).sub(1.0)
    value = model(sr_lp, hr_lp)
    return value.mean().item()


def save_example_strip(path, hr_tensor, lr_tensor, sr_tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    hr_cpu = hr_tensor.detach().cpu()
    lr_cpu = lr_tensor.detach().cpu()
    sr_cpu = sr_tensor.detach().cpu()
    target_size = hr_cpu.shape[-2:]
    lr_nearest = F.interpolate(lr_cpu, size=target_size, mode="nearest")
    lr_bicubic = F.interpolate(lr_cpu, size=target_size, mode="bicubic", align_corners=False)
    strip = torch.cat([hr_cpu, lr_nearest, lr_bicubic, sr_cpu], dim=-1)
    array = strip.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    plt.imsave(path.as_posix(), array)


def write_metrics_csv(path, rows):
    fieldnames = ["image", "psnr", "ssim", "l1", "lpips", "grad_l1"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "psnr": f"{row['psnr']:.6f}",
                    "ssim": "" if row["ssim"] is None else f"{row['ssim']:.6f}",
                    "l1": f"{row['l1']:.6f}",
                    "lpips": "" if row["lpips"] is None else f"{row['lpips']:.6f}",
                    "grad_l1": f"{row['grad_l1']:.6f}",
                }
            )


def write_summary_files(out_dir, summaries):
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    fieldnames = ["label", "psnr", "ssim", "l1", "lpips", "grad_l1", "count"]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            ssim_str = "" if math.isnan(row["ssim"]) else f"{row['ssim']:.6f}"
            writer.writerow(
                {
                    "label": row["label"],
                    "psnr": f"{row['psnr']:.6f}",
                    "ssim": ssim_str,
                    "l1": f"{row['l1']:.6f}",
                    "lpips": "" if math.isnan(row["lpips"]) else f"{row['lpips']:.6f}",
                    "grad_l1": f"{row['grad_l1']:.6f}",
                    "count": row["count"],
                }
            )
    payload = {"models": []}
    for row in summaries:
        payload["models"].append(
            {
                "label": row["label"],
                "psnr": round(row["psnr"], 6),
                "ssim": None if math.isnan(row["ssim"]) else round(row["ssim"], 6),
                "l1": round(row["l1"], 6),
                "lpips": None if math.isnan(row["lpips"]) else round(row["lpips"], 6),
                "grad_l1": round(row["grad_l1"], 6),
                "count": row["count"],
            }
        )
    summary_json.write_text(json.dumps(payload, indent=2))


def make_bar_chart(out_path, summaries, key, title, ylabel):
    if not summaries:
        return
    labels = [row["label"] for row in summaries]
    values = []
    label_values = []
    for row in summaries:
        value = row[key]
        if math.isnan(value):
            values.append(0.0)
            label_values.append(float("nan"))
        else:
            values.append(value)
            label_values.append(value)
    width = max(4.0, len(labels) * 1.4)
    fig, ax = plt.subplots(figsize=(width, 4.0))
    indices = range(len(labels))
    bars = ax.bar(indices, values, color="#4c72b0")
    ax.set_xticks(list(indices))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ymax = max(values) if values else 0.0
    if ymax <= 0.0:
        ax.set_ylim(0.0, 1.0)
    for bar, value in zip(bars, label_values):
        height = bar.get_height()
        text = "nan" if isinstance(value, float) and math.isnan(value) else f"{value:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, text, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=200)
    plt.close(fig)


def evaluate_model(
    label,
    ckpt_path,
    model_cfg,
    degradation_cfg,
    paths,
    device,
    out_root,
    use_y,
    crop_border,
    dump_count,
    lpips_model,
    mode="normal",
    tta_enabled=True,
    variant="model",
):
    state = _safe_torch_load(ckpt_path)
    ckpt_model_cfg = None
    if isinstance(state, dict):
        cfg = state.get("config")
        if isinstance(cfg, dict):
            ckpt_model_cfg = cfg.get("model")

    model_cfg_local = ckpt_model_cfg or model_cfg
    residual_enabled = bool(model_cfg_local.get("residual_output", False))
    model = build_model(model_cfg_local, device)
    load_checkpoint(model, ckpt_path, variant, state=state)

    model_dir = (Path(out_root) / label)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows, psnr_values, ssim_values = [], [], []
    l1_values, lpips_values, grad_values = [], [], []
    net_scale = int(model_cfg_local["scale"])
    if mode == "twopass" and net_scale != 2:
        raise ValueError(f"twopass expects a ×2 model, got ×{net_scale}")
    degrade_scale = 4 if mode == "twopass" else net_scale

    with torch.inference_mode():
        for idx, path in enumerate(paths):
            hr_image = load_image(path)
            rng = random.Random(idx)
            lr = downscale_with_profile(hr_image, degrade_scale, degradation_cfg, rng).unsqueeze(0).to(device)
            hr = tensor_from_pil(hr_image).unsqueeze(0).to(device)

            sr = run_inference(model, lr, net_scale, residual_enabled, tta_enabled, mode)

            sr_aligned, hr_aligned = center_crop_to_match(sr, hr)
            sr_eval = crop_border_tensor(sr_aligned, crop_border)
            hr_eval = crop_border_tensor(hr_aligned, crop_border)

            sr_metric = to_y_channel(sr_eval) if use_y else sr_eval
            hr_metric = to_y_channel(hr_eval) if use_y else hr_eval

            p = compute_psnr(sr_metric, hr_metric)
            s = compute_ssim(sr_metric, hr_metric)
            l1 = compute_l1(sr_metric, hr_metric)
            lp = compute_lpips(sr_eval, hr_eval, lpips_model)
            g = compute_grad_l1(sr_eval, hr_eval)

            metrics_rows.append({"image": str(path), "psnr": p, "ssim": s, "l1": l1, "lpips": lp, "grad_l1": g})
            psnr_values.append(p)
            if s is not None:
                ssim_values.append(s)
            l1_values.append(l1)
            if lp is not None:
                lpips_values.append(lp)
            grad_values.append(g)

            if idx < dump_count:
                save_example_strip(model_dir / "examples" / f"{idx:04d}.png", hr_aligned, lr, sr_aligned)

    write_metrics_csv(model_dir / "metrics.csv", metrics_rows)
    mean_psnr = sum(psnr_values) / len(psnr_values)
    mean_ssim = float("nan") if not ssim_values else sum(ssim_values) / len(ssim_values)
    mean_l1 = sum(l1_values) / len(l1_values)
    mean_lpips = float("nan") if not lpips_values else sum(lpips_values) / len(lpips_values)
    mean_grad = sum(grad_values) / len(grad_values)
    return {
        "label": label,
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "l1": mean_l1,
        "lpips": mean_lpips,
        "grad_l1": mean_grad,
        "count": len(paths),
    }

def main():
    args = parse_args()
    dump_count = max(0, args.dump)
    crop_border = max(0, args.crop)
    config = load_config(args.config)
    model_cfg = config["model"]
    degradation_cfg = DegradationConfig(**config["degradations"])
    paths = resolve_eval_paths(config)
    device = pick_device(args.device)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    model_specs = parse_model_specs(args.models)
    lpips_model = None if args.no_lpips else build_lpips(args.lpips_net, device)
    eval_cfg = config.get("eval", {})
    tta_enabled = bool(eval_cfg.get("tta", True))
    summaries = []
    for label, ckpt_path, mode, variant in model_specs:
        summary = evaluate_model(
            label=label,
            ckpt_path=ckpt_path,
            model_cfg=model_cfg,
            degradation_cfg=degradation_cfg,
            paths=paths,
            device=device,
            out_root=out_root,
            use_y=args.y,
            crop_border=crop_border,
            dump_count=dump_count,
            lpips_model=lpips_model,
            mode=mode,
            tta_enabled=tta_enabled,
            variant=variant,
        )
        summaries.append(summary)
    write_summary_files(out_root, summaries)
    make_bar_chart(out_root / "bar_psnr.png", summaries, "psnr", "PSNR", "PSNR (dB)")
    make_bar_chart(out_root / "bar_ssim.png", summaries, "ssim", "SSIM", "SSIM")
    make_bar_chart(out_root / "bar_l1.png", summaries, "l1", "L1", "L1 (lower better)")
    make_bar_chart(out_root / "bar_lpips.png", summaries, "lpips", "LPIPS", "LPIPS (lower better)")


if __name__ == "__main__":
    main()
