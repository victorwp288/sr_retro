#!/usr/bin/env python3

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SOBEL_KERNEL_X = torch.tensor(
    [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32
).view(1, 1, 3, 3)
SOBEL_KERNEL_Y = SOBEL_KERNEL_X.transpose(2, 3).contiguous()
_KERNELS = None


class Config:
    __slots__ = (
        "split_dir",
        "base_dir",
        "source_root",
        "output_root",
        "patch_sizes",
        "top_k",
        "tile_step",
        "only_train",
        "workers",
        "device",
    )

    def __init__(self):
        parser = argparse.ArgumentParser(description="Precompute Sobel-based edge metadata.")
        parser.add_argument("--split-dir", required=True, help="Directory containing train/val/test txt files.")
        parser.add_argument("--base-dir", default=".", help="Directory used to resolve relative split paths.")
        parser.add_argument("--source-root", required=True, help="Root that contains all referenced images.")
        parser.add_argument(
            "--output-root",
            required=True,
            help="Directory where metadata JSON files will be stored (mirrors source tree).",
        )
        parser.add_argument("--patch-sizes", default="64,96,128,160", help="Comma-separated list of patch sizes.")
        parser.add_argument("--top-k", type=int, default=16, help="Number of candidates to keep per patch size.")
        parser.add_argument("--tile-step", type=int, default=1, help="Stride between candidate patches.")
        parser.add_argument(
            "--only-train",
            action="store_true",
            help="Only process training split paths (still writes val/test splits unchanged).",
        )
        parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 = os.cpu_count()).")
        parser.add_argument("--device", default="cpu", help="Torch device for Sobel convs (e.g., 'cpu', 'cuda:0').")
        args = parser.parse_args()

        patch_sizes = [int(item) for item in args.patch_sizes.split(",") if item.strip()]
        if not patch_sizes:
            raise ValueError("--patch-sizes must include at least one value")

        self.split_dir = Path(args.split_dir)
        self.base_dir = Path(args.base_dir)
        self.source_root = Path(args.source_root).resolve()
        self.output_root = Path(args.output_root).resolve()
        self.patch_sizes = tuple(patch_sizes)
        self.top_k = args.top_k
        self.tile_step = max(1, args.tile_step)
        self.only_train = args.only_train
        self.workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
        self.device = args.device


def read_split_file(path):
    lines = []
    for raw in path.read_text().splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        lines.append(Path(entry))
    return lines


def ensure_under_root(path, root):
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:  # pragma: no cover - user guidance
        raise ValueError(f"{path} is not inside source root {root}") from exc


def pil_to_tensor(image):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr) / 255.0


def compute_sobel_magnitude(tensor):
    gray = tensor.mean(dim=0, keepdim=True).unsqueeze(0)
    grad_x = F.conv2d(gray, SOBEL_KERNEL_X.to(tensor.device), padding=1)
    grad_y = F.conv2d(gray, SOBEL_KERNEL_Y.to(tensor.device), padding=1)
    return torch.abs(grad_x) + torch.abs(grad_y)


def topk_candidates(magnitude, patch_size, stride, top_k):
    height = int(magnitude.shape[-2])
    width = int(magnitude.shape[-1])
    if height < patch_size or width < patch_size:
        return []
    pooled = F.avg_pool2d(magnitude, kernel_size=patch_size, stride=stride)
    flat = pooled.flatten()
    if flat.numel() == 0:
        return []
    count = min(top_k, flat.numel())
    values, indices = torch.topk(flat, count)
    num_cols = pooled.shape[-1]
    seen = set()
    entries = []
    for val, idx in zip(values.tolist(), indices.tolist()):
        row = idx // num_cols
        col = idx % num_cols
        x = int(col * stride)
        y = int(row * stride)
        if x + patch_size > width:
            x = max(0, width - patch_size)
        if y + patch_size > height:
            y = max(0, height - patch_size)
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        entries.append({"x": x, "y": y, "score": float(val)})
    return entries


def collect_paths(split_dir, base_dir, only_train):
    results = []
    for name in ("train", "val", "test"):
        if only_train and name != "train":
            continue
        split_path = split_dir / f"{name}.txt"
        if not split_path.exists():
            continue
        entries = read_split_file(split_path)
        for entry in entries:
            abs_path = entry if entry.is_absolute() else (base_dir / entry)
            results.append(abs_path.resolve())
    return sorted(set(results))


def build_tasks(paths, source_root, output_root, patch_sizes, top_k, tile_step):
    tasks = []
    skipped = 0
    for path in paths:
        try:
            rel_path = ensure_under_root(path, source_root)
        except ValueError:
            continue
        out_path = (output_root / rel_path).with_suffix(rel_path.suffix + ".edge.json")
        if out_path.exists():
            skipped += 1
            continue
        tasks.append((str(path), str(rel_path), str(output_root), patch_sizes, top_k, tile_step))
    return tasks, skipped


def _prepare_sobel_kernels(device):
    return (
        SOBEL_KERNEL_X.to(device=device),
        SOBEL_KERNEL_Y.to(device=device),
    )


def _process_init(device_str):
    global _KERNELS
    device = torch.device(device_str)
    _KERNELS = _prepare_sobel_kernels(device)


def _process_image(task):
    path_str, rel_str, output_root_str, patch_sizes, top_k, tile_step = task
    path = Path(path_str)
    rel_path = Path(rel_str)
    output_root = Path(output_root_str)
    out_path = (output_root / rel_path).with_suffix(rel_path.suffix + ".edge.json")
    if out_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        tensor = pil_to_tensor(img.convert("RGB"))
    kernels = _KERNELS or _prepare_sobel_kernels(torch.device("cpu"))
    device = kernels[0].device
    magnitude = compute_sobel_magnitude(tensor.to(device))
    candidates = {}
    for size in patch_sizes:
        entries = topk_candidates(magnitude, size, tile_step, top_k)
        if entries:
            candidates[str(size)] = entries
    payload = {
        "width": tensor.shape[-1],
        "height": tensor.shape[-2],
        "patch_candidates": candidates,
    }
    out_path.write_text(json.dumps(payload))
    return True


def execute_tasks(tasks, workers, device):
    processed = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_process_init,
        initargs=(device,),
    ) as executor:
        futures = [executor.submit(_process_image, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="edge metadata"):
            if future.result():
                processed += 1
    return processed


def run(config):
    image_paths = collect_paths(config.split_dir, config.base_dir, config.only_train)
    if not image_paths:
        raise ValueError("No image paths found in the provided split directory")
    tasks, skipped = build_tasks(
        image_paths,
        config.source_root,
        config.output_root,
        config.patch_sizes,
        config.top_k,
        config.tile_step,
    )
    if not tasks:
        print("No new images to process (all metadata up to date).")
        return
    processed = execute_tasks(tasks, config.workers, config.device)
    print(f"Wrote metadata for {processed} images (skipped {skipped} existing) to {config.output_root}")


def main():
    run(Config())


if __name__ == "__main__":
    main()
