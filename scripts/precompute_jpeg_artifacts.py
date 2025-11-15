#!/usr/bin/env python3

import argparse
import io
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SPLIT_NAMES = ("train", "val", "test")
AUGMENT_PLAN = {"train": True, "val": False, "test": False}


class Config:
    __slots__ = (
        "split_dir",
        "base_dir",
        "source_root",
        "output_root",
        "output_split_dir",
        "quality_min",
        "quality_max",
        "suffix",
        "only_train",
        "overwrite",
        "seed",
    )

    def __init__(self):
        parser = argparse.ArgumentParser(description="Precompute JPEG artifact variants per split.")
        parser.add_argument("--split-dir", required=True, help="Directory containing train/val/test txt files.")
        parser.add_argument(
            "--base-dir",
            default=".",
            help="Base directory to resolve relative paths from split files (defaults to repo root).",
        )
        parser.add_argument("--source-root", required=True, help="Common root directory for the original images.")
        parser.add_argument(
            "--output-root",
            required=True,
            help="Directory where baked images will be written (mirrors the source tree).",
        )
        parser.add_argument(
            "--output-split-dir",
            required=True,
            help="Directory to write the updated split files that include the baked images.",
        )
        parser.add_argument("--quality-min", type=int, default=70)
        parser.add_argument("--quality-max", type=int, default=95)
        parser.add_argument("--suffix", default="_jpeg", help="Filename suffix added before the extension.")
        parser.add_argument(
            "--only-train",
            action="store_true",
            help="Only augment the training split; still copies val/test files unchanged.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Regenerate baked images even if the destination file already exists.",
        )
        parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling JPEG quality values.")
        args = parser.parse_args()

        if args.quality_min > args.quality_max:
            raise ValueError("--quality-min must be <= --quality-max")

        self.split_dir = Path(args.split_dir)
        self.base_dir = Path(args.base_dir).resolve()
        self.source_root = Path(args.source_root).resolve()
        self.output_root = Path(args.output_root).resolve()
        self.output_split_dir = Path(args.output_split_dir)
        self.quality_min = args.quality_min
        self.quality_max = args.quality_max
        self.suffix = args.suffix
        self.only_train = args.only_train
        self.overwrite = args.overwrite
        self.seed = args.seed


def read_split_file(path):
    lines = []
    for raw in path.read_text().splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        lines.append(Path(value))
    return lines


def ensure_under_root(path, root):
    try:
        return path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - helpful error for users
        raise ValueError(f"{path} is not inside --source-root {root}") from exc


def jpeg_roundtrip(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    degraded = Image.open(buffer)
    return degraded.convert("RGB")


def bake_image(src_path, dst_path, quality, overwrite):
    if dst_path.exists() and not overwrite:
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        degraded = jpeg_roundtrip(img.convert("RGB"), quality)
        degraded.save(dst_path)


def write_split(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for entry in items:
            handle.write(f"{entry.as_posix()}\n")


def load_split_entries(split_dir, split_name):
    split_path = split_dir / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"missing split file: {split_path}")
    return read_split_file(split_path)


def should_augment(split_name, only_train):
    base_choice = AUGMENT_PLAN[split_name]
    return base_choice or not only_train


def augment_entries(entries, split_name, config):
    if not should_augment(split_name, config.only_train):
        return entries, 0
    augmented = []
    iterator = tqdm(entries, desc=f"baking {split_name}")
    for rel_path in iterator:
        abs_path = rel_path if rel_path.is_absolute() else (config.base_dir / rel_path)
        abs_path = abs_path.resolve()
        rel_to_root = ensure_under_root(abs_path, config.source_root)
        aug_name = rel_to_root.with_name(f"{rel_to_root.stem}{config.suffix}{rel_to_root.suffix}")
        dst_path = config.output_root / aug_name
        quality = random.randint(config.quality_min, config.quality_max)
        bake_image(abs_path, dst_path, quality, config.overwrite)
        try:
            rel_record = dst_path.relative_to(config.base_dir)
        except ValueError:
            rel_record = dst_path
        augmented.append(rel_record)
    return entries + augmented, len(augmented)


def run(config):
    random.seed(config.seed)
    processed_counts = {name: 0 for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        entries = load_split_entries(config.split_dir, split_name)
        combined, baked = augment_entries(entries, split_name, config)
        processed_counts[split_name] = baked
        out_split_path = config.output_split_dir / f"{split_name}.txt"
        write_split(out_split_path, combined)
    print("JPEG baking summary:")
    for split_name in SPLIT_NAMES:
        print(f"  {split_name:>5}: baked {processed_counts[split_name]} images")
    print(f"Updated split files written to {config.output_split_dir}")


def main():
    run(Config())


if __name__ == "__main__":
    main()
