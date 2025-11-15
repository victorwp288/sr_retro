#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def collect_groups(root):
    groups = {}
    misc = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = None
        stem = path.stem
        if stem.startswith("game"):
            part = stem.split("_", 1)[0]
            if part[4:].isdigit():
                key = part
        if key is None:
            misc.append(path)
        else:
            groups.setdefault(key, []).append(path)
    if misc:
        groups["misc"] = misc
    return groups


def split_group(paths, ratios=(0.8, 0.1, 0.1)):
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = paths[:n_train]
    val = paths[n_train : n_train + n_val]
    test = paths[n_train + n_val :]
    return train, val, test


def write_list(out_path, items):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for item in items:
            f.write(f"{item.as_posix()}\n")


def main():
    parser = argparse.ArgumentParser(description="Stratify dataset by gameX prefixes.")
    parser.add_argument("--root", required=True, help="Directory with frames/tiles.")
    parser.add_argument(
        "--output_dir",
        default="splits",
        help="Where to write train/val/test txt files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    if ratios[2] < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1")

    root = Path(args.root)
    groups = collect_groups(root)

    train_all, val_all, test_all = [], [], []
    for key, paths in groups.items():
        t, v, te = split_group(paths, ratios)
        train_all.extend(t)
        val_all.extend(v)
        test_all.extend(te)
        print(f"{key:>6}: train {len(t)} | val {len(v)} | test {len(te)}")

    out_dir = Path(args.output_dir)
    write_list(out_dir / "train.txt", train_all)
    write_list(out_dir / "val.txt", val_all)
    write_list(out_dir / "test.txt", test_all)
    print(
        f"Wrote {len(train_all)} train / {len(val_all)} val / {len(test_all)} test paths to {out_dir}"
    )


if __name__ == "__main__":
    main()
