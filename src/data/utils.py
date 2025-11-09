import random
import threading
from pathlib import Path

from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga"}
_AUTO_SPLIT_CACHE = {}
_AUTO_SPLIT_CACHE_LOCK = threading.Lock()


def load_image(path):
    return Image.open(path).convert("RGB")


def choose_hr_crop(image, patch_size, scale, tile_aligned, rng):
    if patch_size is None:
        return image
    width, height = image.size
    if patch_size > width and patch_size > height:
        # Upsample only when both dimensions are smaller than the requested patch.
        return ImageOps.fit(image, (patch_size, patch_size), method=Image.NEAREST, centering=(0.5, 0.5))
    if tile_aligned:
        step = max(scale, 1)
        max_x = width - patch_size
        max_y = height - patch_size
        x = 0 if max_x <= 0 else rng.randrange(0, max_x + 1, step)
        y = 0 if max_y <= 0 else rng.randrange(0, max_y + 1, step)
    else:
        max_x = width - patch_size
        max_y = height - patch_size
        x = 0 if max_x <= 0 else rng.randint(0, max_x)
        y = 0 if max_y <= 0 else rng.randint(0, max_y)
    return image.crop((x, y, x + patch_size, y + patch_size))


def apply_augment(image, allow_flips, allow_rotation, rng):
    if allow_flips and rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if allow_flips and rng.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    if allow_rotation and rng.random() < 0.5:
        image = image.transpose(Image.ROTATE_180)
    return image


def tensor_from_pil(image):
    return to_tensor(image).float()


def list_image_paths(root):
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"data root {root} does not exist")
    candidates = []
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._"):
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"no image files found in {root}")
    candidates.sort()
    return [p.as_posix() for p in candidates]


def read_split_file(path):
    lines = Path(path).read_text().splitlines()
    items = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not items:
        raise RuntimeError(f"split file {path} is empty")
    return items


def auto_split_paths(root, ratios, seed):
    key = (Path(root).resolve(), tuple(sorted(ratios.items())), seed)
    with _AUTO_SPLIT_CACHE_LOCK:
        cached = _AUTO_SPLIT_CACHE.get(key)
    if cached is not None:
        return cached

    paths = list_image_paths(root)
    total = len(paths)
    rng = random.Random(seed)
    order = paths[:]
    rng.shuffle(order)

    ratio_map = {
        "train": ratios.get("train", 0.8),
        "val": ratios.get("val", 0.1),
    }
    ratio_map["test"] = ratios.get("test", max(0.0, 1.0 - ratio_map["train"] - ratio_map["val"]))
    ratio_map = {key_: max(0.0, value) for key_, value in ratio_map.items()}

    subsets = ["train", "val", "test"]
    counts = {}
    fractions = []
    for subset in subsets:
        raw = ratio_map[subset] * total
        count = int(raw)
        counts[subset] = count
        fractions.append((raw - count, subset))

    allocated = sum(counts.values())
    remaining = total - allocated
    positive_targets = [s for s in subsets if ratio_map[s] > 0]
    if not positive_targets:
        positive_targets = subsets

    if remaining > 0:
        fractions.sort(reverse=True)
        idx = 0
        targets = positive_targets or subsets
        while remaining > 0:
            subset = fractions[idx % len(fractions)][1]
            if subset not in targets:
                subset = targets[idx % len(targets)]
            counts[subset] += 1
            remaining -= 1
            idx += 1
    elif remaining < 0:
        while remaining < 0:
            subset = max(subsets, key=lambda s: counts[s])
            if counts[subset] == 0:
                break
            counts[subset] -= 1
            remaining += 1

    for subset in subsets:
        expected = ratio_map.get(subset, 0.0) * total
        if expected >= 1.0 and counts[subset] == 0:
            donor_candidates = [s for s in subsets if s != subset and counts[s] > 1]
            if not donor_candidates:
                donor_candidates = [s for s in subsets if s != subset and counts[s] > 0]
            if donor_candidates:
                donor = max(donor_candidates, key=lambda s: counts[s])
                counts[donor] -= 1
                counts[subset] += 1

    assigned = sum(counts.values())
    if assigned < total:
        diff = total - assigned
        targets = positive_targets or subsets
        for _ in range(diff):
            subset = max(targets, key=lambda s: ratio_map.get(s, 0.0))
            counts[subset] += 1
    elif assigned > total:
        diff = assigned - total
        for _ in range(diff):
            subset = max(subsets, key=lambda s: counts[s])
            if counts[subset] == 0:
                continue
            counts[subset] -= 1

    idx = 0
    train = order[idx : idx + counts["train"]]
    idx += counts["train"]
    val = order[idx : idx + counts["val"]]
    idx += counts["val"]
    test = order[idx : idx + counts["test"]]

    splits = {
        "train": train,
        "val": val,
        "test": test,
        "all": order,
    }
    with _AUTO_SPLIT_CACHE_LOCK:
        cached = _AUTO_SPLIT_CACHE.get(key)
        if cached is not None:
            return cached
        _AUTO_SPLIT_CACHE[key] = splits
    return splits


def resolve_paths_from_source(source, subset=None, ratios=None, seed=42):
    if source is None:
        raise RuntimeError("no data source provided")
    path = Path(source)
    if path.is_file():
        items = read_split_file(path)
        return items
    if path.is_dir():
        ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
        subset = subset or "all"
        splits = auto_split_paths(path, ratios, seed)
        if subset not in splits:
            raise RuntimeError(f"subset {subset} not available for auto split")
        return splits[subset]
    raise RuntimeError(f"data source {source} is neither a file nor directory")
