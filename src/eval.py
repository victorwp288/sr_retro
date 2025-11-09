import argparse
import csv
import json
import math
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from src.data.degradations import DegradationConfig, downscale_with_profile
from src.data.utils import load_image, tensor_from_pil, resolve_paths_from_source
from src.models import EDSR


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
        mode = "normal"
        if ckpt.startswith("twopass:"):
            mode = "twopass"
            ckpt = ckpt[len("twopass:"):]
        if not label:
            label = Path(ckpt).stem
        specs.append((label, Path(ckpt), mode))
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


def load_checkpoint(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = _select_state_dict(state)
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


def to_y_channel(tensor):
    if tensor.size(1) == 1:
        return tensor
    if tensor.size(1) != 3:
        return tensor
    r = tensor[:, 0:1]
    g = tensor[:, 1:2]
    b = tensor[:, 2:3]
    return 0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0


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


def save_example_strip(path, hr_tensor, lr_tensor, sr_tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    hr_cpu = hr_tensor.detach().cpu()
    lr_cpu = lr_tensor.detach().cpu()
    sr_cpu = sr_tensor.detach().cpu()
    target_size = hr_cpu.shape[-2:]
    lr_up = F.interpolate(lr_cpu, size=target_size, mode="bicubic", align_corners=False)
    strip = torch.cat([hr_cpu, lr_up, sr_cpu], dim=-1)
    array = strip.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    plt.imsave(path.as_posix(), array)


def write_metrics_csv(path, rows):
    fieldnames = ["image", "psnr", "ssim"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "psnr": f"{row['psnr']:.6f}",
                    "ssim": "" if row["ssim"] is None else f"{row['ssim']:.6f}",
                }
            )


def write_summary_files(out_dir, summaries):
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    fieldnames = ["label", "psnr", "ssim", "count"]
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


def evaluate_model(label, ckpt_path, model_cfg, degradation_cfg, paths, device,
                   out_root, use_y, crop_border, dump_count, mode="normal"):

    # load ckpt/config (unchanged) ...
    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_model_cfg = None
    if isinstance(state, dict):
        cfg = state.get("config")
        if isinstance(cfg, dict):
            ckpt_model_cfg = cfg.get("model")

    model_cfg_local = ckpt_model_cfg or model_cfg
    model = build_model(model_cfg_local, device)  # this will be x2 for base, x4 for x4 ckpts

    state_dict = _select_state_dict(state)
    if hasattr(state_dict, "items"):
        state_dict = _clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)

    model_dir = (Path(out_root) / label)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows, psnr_values, ssim_values = [], [], []
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

            if mode == "twopass":
                # x2 model run twice to get x4
                mid = model(lr).clamp(0.0, 1.0)
                sr  = model(mid).clamp(0.0, 1.0)
            else:
                sr  = model(lr).clamp(0.0, 1.0)

            sr_aligned, hr_aligned = center_crop_to_match(sr, hr)
            sr_eval = crop_border_tensor(sr_aligned, crop_border)
            hr_eval = crop_border_tensor(hr_aligned, crop_border)

            sr_metric = to_y_channel(sr_eval) if use_y else sr_eval
            hr_metric = to_y_channel(hr_eval) if use_y else hr_eval

            p = compute_psnr(sr_metric, hr_metric)
            s = compute_ssim(sr_metric, hr_metric)

            metrics_rows.append({"image": str(path), "psnr": p, "ssim": s})
            psnr_values.append(p)
            if s is not None:
                ssim_values.append(s)

            if idx < dump_count:
                save_example_strip(model_dir / "examples" / f"{idx:04d}.png", hr_aligned, lr, sr_aligned)

    write_metrics_csv(model_dir / "metrics.csv", metrics_rows)
    mean_psnr = sum(psnr_values) / len(psnr_values)
    mean_ssim = float("nan") if not ssim_values else sum(ssim_values) / len(ssim_values)
    return {"label": label, "psnr": mean_psnr, "ssim": mean_ssim, "count": len(paths)}

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
    summaries = []
    for label, ckpt_path, mode in model_specs:
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
            mode=mode,   # <- add this
        )
        summaries.append(summary)
    write_summary_files(out_root, summaries)
    make_bar_chart(out_root / "bar_psnr.png", summaries, "psnr", "PSNR", "PSNR (dB)")
    make_bar_chart(out_root / "bar_ssim.png", summaries, "ssim", "SSIM", "SSIM")


if __name__ == "__main__":
    main()
