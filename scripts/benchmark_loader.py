import argparse
import time
from pathlib import Path

import torch

from src.data import PixelArtDataset, DegradationConfig
from src.data.utils import auto_split_paths, read_split_file
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    data_cfg = config["data"]
    model_cfg = config["model"]
    degradation = DegradationConfig(**config["degradations"])

    if data_cfg.get("root"):
        splits = auto_split_paths(
            data_cfg["root"],
            data_cfg.get("auto_split_ratio", {"train": 0.8, "val": 0.1, "test": 0.1}),
            data_cfg.get("auto_split_seed", config.get("seed", 42)),
        )
        train_paths = splits["train"]
    else:
        train_source = data_cfg.get("train_split")
        if not train_source:
            raise RuntimeError("benchmark requires data.root or data.train_split")
        train_paths = read_split_file(train_source)

    dataset = PixelArtDataset(
        paths=train_paths,
        scale=model_cfg["scale"],
        patch_size_hr=data_cfg["patch_size_hr"],
        degradation_config=degradation,
        crops_tile_aligned_prob=data_cfg["crops_tile_aligned_prob"],
        flips=data_cfg["flips"],
        rotations=data_cfg["rotations"],
        augment=True,
        cache_images=data_cfg.get("cache_images", True),
    )

    train_workers = data_cfg.get("train_workers", 4)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": config["training"]["batch_size"],
        "shuffle": True,
        "num_workers": train_workers,
    }
    if train_workers > 0:
        loader_kwargs["prefetch_factor"] = data_cfg.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = data_cfg.get("persistent_workers", True)
    else:
        loader_kwargs["persistent_workers"] = False

    loader = torch.utils.data.DataLoader(**loader_kwargs)

    it = iter(loader)
    timings = []
    total = args.warmup + args.steps
    for step in range(total):
        start = time.perf_counter()
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)
        end = time.perf_counter()
        if step >= args.warmup:
            timings.append(end - start)
    avg = sum(timings) / len(timings)
    print(f"Average batch time over {args.steps} iterations: {avg * 1000:.2f} ms")


if __name__ == "__main__":
    main()
