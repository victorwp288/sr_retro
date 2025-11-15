import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


from .utils import (
    load_image,
    choose_hr_crop,
    apply_augment,
    tensor_from_pil,
)
from .degradations import downscale_tensor_with_profile

_SOBEL_KERNEL_X = torch.tensor(
    [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)
_SOBEL_KERNEL_Y = _SOBEL_KERNEL_X.transpose(2, 3).contiguous()

_IMAGE_CACHE_MAX_ITEMS = max(1, int(os.environ.get("SR_RETRO_IMAGE_CACHE_SIZE", "32")))


class _ThreadSafeLRUCache:
    def __init__(self, max_items):
        self._max_items = max(1, int(max_items))
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def get_or_set(self, key, loader):
        value = self.get(key)
        if value is not None:
            return value
        loaded = loader()
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
            self._entries[key] = loaded
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_items:
                self._entries.popitem(last=False)
            return loaded


_IMAGE_CACHE = _ThreadSafeLRUCache(_IMAGE_CACHE_MAX_ITEMS)


def _sobel_score(tensor):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    gray = tensor.mean(dim=1, keepdim=True)
    device = gray.device
    dtype = gray.dtype
    grad_x = F.conv2d(gray, _SOBEL_KERNEL_X.to(device=device, dtype=dtype), padding=1)
    grad_y = F.conv2d(gray, _SOBEL_KERNEL_Y.to(device=device, dtype=dtype), padding=1)
    magnitude = torch.abs(grad_x) + torch.abs(grad_y)
    return float(magnitude.mean())


class _TorchRandomAdapter:
    def __init__(self, seed):
        self._generator = torch.Generator()
        self._generator.manual_seed(int(seed))

    def random(self):
        return torch.rand((), generator=self._generator).item()

    def randint(self, a, b):
        if b < a:
            raise ValueError("randint() upper bound must be >= lower bound")
        # torch.randint is exclusive on the upper bound.
        return int(torch.randint(a, b + 1, (1,), generator=self._generator).item())

    def randrange(self, start, stop=None, step=1):
        if step <= 0:
            raise ValueError("randrange() step must be > 0")
        if stop is None:
            start, stop = 0, start
        width = stop - start
        if width <= 0:
            raise ValueError("empty range for randrange()")
        choices = (width + step - 1) // step
        offset = torch.randint(0, choices, (1,), generator=self._generator).item()
        return int(start + step * offset)


class PixelArtDataset(Dataset):
    def __init__(
        self,
        paths,
        scale,
        patch_size_hr,
        degradation_config,
        crops_tile_aligned_prob,
        flips,
        rotations,
        augment=True,
        cache_images=True,
        patch_size_ref=None,
        degradation_configs=None,
        degradation_phase_ref=None,
        edge_sampling=None,
        defer_degradation=False,
        edge_metadata_root=None,
        edge_metadata_source_root=None,
        pt_cache_root=None,
        pt_cache_source_root=None,
    ):
        if not paths:
            raise RuntimeError("dataset received empty path list")
        self.paths = paths
        self.scale = scale
        self.patch_size_hr = patch_size_hr
        self.patch_size_ref = patch_size_ref
        self.degradation_config = degradation_config
        self.degradation_configs = degradation_configs or [degradation_config]
        self.degradation_phase_ref = degradation_phase_ref
        self.crops_tile_aligned_prob = crops_tile_aligned_prob
        self.flips = flips
        self.rotations = rotations
        self.augment = augment
        self.cache_images = cache_images
        self.defer_degradation = bool(defer_degradation)
        self._cache_key = None
        self.images = None
        if cache_images:
            self._cache_key = self._compute_cache_key(self.paths)
            self._bind_cached_images()
        self._rngs = {}
        edge_cfg = edge_sampling or {}
        self.edge_sampling_enabled = bool(edge_cfg.get("enabled"))
        self.edge_candidates = max(1, int(edge_cfg.get("candidates", 1)))
        self.edge_prefer_top = float(edge_cfg.get("prefer_top_p", 0.5))
        self.edge_metadata_root = Path(edge_metadata_root).resolve() if edge_metadata_root else None
        self.edge_metadata_source_root = (
            Path(edge_metadata_source_root).resolve()
            if edge_metadata_source_root
            else None
        )
        self._edge_metadata_cache = {}
        self._edge_metadata_missing = set()
        self.pt_cache_root = Path(pt_cache_root).resolve() if pt_cache_root else None
        self.pt_cache_source_root = (
            Path(pt_cache_source_root).resolve() if pt_cache_source_root else None
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rng = self._resolve_rng()
        if self.images is not None:
            hr = self.images[index].copy()
        else:
            hr = self._load_image(index)
        tile_aligned = rng.random() < self.crops_tile_aligned_prob
        patch_size = self._current_patch_size()
        if (
            tile_aligned
            and self.edge_sampling_enabled
            and patch_size is not None
        ):
            metadata = self._edge_metadata_for_index(index)
        else:
            metadata = None
        if metadata is not None:
            patch = self._pick_patch_from_metadata(hr, metadata, patch_size, rng)
            if patch is not None:
                hr = patch
            else:
                hr = self._pick_patch(hr, tile_aligned, rng)
        else:
            hr = self._pick_patch(hr, tile_aligned, rng)
        if self.augment:
            hr = apply_augment(hr, self.flips, self.rotations, rng)
        hr_tensor = tensor_from_pil(hr)
        sample_seed = rng.randint(0, 2**31 - 1)
        if self.defer_degradation:
            return hr_tensor, sample_seed
        local_rng = random.Random(sample_seed)
        lr_tensor = downscale_tensor_with_profile(
            hr_tensor,
            self.scale,
            self._active_degradation_config(),
            local_rng,
        )
        return lr_tensor, hr_tensor

    def _current_patch_size(self):
        if self.patch_size_ref is not None:
            return int(self.patch_size_ref.value)
        return self.patch_size_hr

    def _active_degradation_config(self):
        if not self.degradation_configs:
            return self.degradation_config
        if self.degradation_phase_ref is None:
            return self.degradation_configs[0]
        idx = int(self.degradation_phase_ref.value)
        idx = max(0, min(idx, len(self.degradation_configs) - 1))
        return self.degradation_configs[idx]

    def _pick_patch(self, image, tile_aligned, rng):
        patch_size = self._current_patch_size()
        if patch_size is None:
            return image
        if not self.edge_sampling_enabled or self.edge_candidates <= 1:
            return choose_hr_crop(image, patch_size, self.scale, tile_aligned, rng)
        best_score = None
        best_crop = None
        other_crops = []
        for _ in range(self.edge_candidates):
            candidate = choose_hr_crop(image, patch_size, self.scale, tile_aligned, rng)
            tensor = tensor_from_pil(candidate).float()
            score = _sobel_score(tensor)
            if best_score is None or score > best_score:
                if best_crop is not None:
                    other_crops.append((best_score, best_crop))
                best_score = score
                best_crop = candidate
            else:
                other_crops.append((score, candidate))
        if best_crop is None:
            fallback = choose_hr_crop(image, patch_size, self.scale, tile_aligned, rng)
            return fallback
        if not other_crops or rng.random() < self.edge_prefer_top:
            return best_crop
        choice = rng.randrange(0, len(other_crops))
        _, crop = other_crops[choice]
        return crop

    def _pick_patch_from_metadata(self, image, metadata, patch_size, rng):
        candidates = (metadata.get("patch_candidates") or {}).get(str(patch_size))
        if not candidates:
            return None
        choice_idx = 0
        if len(candidates) > 1 and rng.random() >= self.edge_prefer_top:
            choice_idx = rng.randint(0, len(candidates) - 1)
        candidate = candidates[choice_idx]
        x = int(candidate.get("x", 0))
        y = int(candidate.get("y", 0))
        patch = image.crop((x, y, x + patch_size, y + patch_size))
        if patch.size != (patch_size, patch_size):
            patch = patch.resize((patch_size, patch_size), Image.NEAREST)
        return patch

    def _resolve_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else None
        rng = self._rngs.get(worker_id)
        if rng is None:
            base_seed = torch.initial_seed()
            seed = base_seed % 2**32 if worker_id is None else (base_seed + worker_id + 1) % 2**32
            rng = _TorchRandomAdapter(seed)
            self._rngs[worker_id] = rng
        return rng

    def _edge_metadata_for_index(self, index):
        if self.edge_metadata_root is None and self.edge_metadata_source_root is None:
            return self._load_inline_metadata(index)
        path = Path(self.paths[index])
        key = str(path.resolve())
        if key in self._edge_metadata_cache:
            return self._edge_metadata_cache[key]
        meta_path = self._locate_edge_metadata(path)
        if meta_path is None:
            self._edge_metadata_cache[key] = None
            return None
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            data = None
        self._edge_metadata_cache[key] = data
        return data

    def _load_inline_metadata(self, index):
        path = Path(self.paths[index])
        candidate = path.with_suffix(path.suffix + ".edge.json")
        key = str(candidate)
        if key in self._edge_metadata_cache:
            return self._edge_metadata_cache[key]
        if not candidate.exists():
            self._edge_metadata_cache[key] = None
            return None
        try:
            data = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            data = None
        self._edge_metadata_cache[key] = data
        return data

    def _locate_edge_metadata(self, image_path: Path):
        candidates = []
        if self.edge_metadata_root is not None:
            rel = None
            if self.edge_metadata_source_root is not None:
                try:
                    rel = image_path.resolve().relative_to(self.edge_metadata_source_root)
                except ValueError:
                    rel = None
            if rel is not None:
                candidates.append((self.edge_metadata_root / rel).with_suffix(rel.suffix + ".edge.json"))
        candidates.append(image_path.with_suffix(image_path.suffix + ".edge.json"))
        for candidate in candidates:
            if candidate in self._edge_metadata_missing:
                continue
            if candidate.exists():
                return candidate
        self._edge_metadata_missing.update(candidates)
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("cache_images"):
            state["images"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.cache_images:
            self._bind_cached_images()
        self._edge_metadata_cache = {}
        self._edge_metadata_missing = set()

    @staticmethod
    def _compute_cache_key(paths):
        digest = hashlib.sha1()
        for path in paths:
            digest.update(str(path).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _bind_cached_images(self):
        if not self._cache_key:
            self._cache_key = self._compute_cache_key(self.paths)
        self.images = _IMAGE_CACHE.get_or_set(
            self._cache_key,
            lambda: [load_image(path) for path in self.paths],
        )

    def _load_image(self, index):
        if self.pt_cache_root is None:
            return load_image(self.paths[index])
        cache_path = self._resolve_pt_cache_path(index)
        if cache_path is None:
            return load_image(self.paths[index])
        if cache_path.is_file():
            tensor = torch.load(cache_path)
            array = tensor.numpy().astype(np.uint8)
            return Image.fromarray(array, mode="RGB")
        image = load_image(self.paths[index])
        array = np.asarray(image, dtype=np.uint8)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(array), cache_path)
        return image

    def _resolve_pt_cache_path(self, index):
        path = Path(self.paths[index]).resolve()
        if self.pt_cache_root is None:
            return path.with_suffix(path.suffix + ".pt")

        rel = None
        if self.pt_cache_source_root is not None:
            try:
                rel = path.relative_to(self.pt_cache_source_root)
            except ValueError:
                rel = None

        if rel is None:
            digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
            rel = Path("_misc") / digest[:2] / f"{path.stem}_{digest[2:10]}"

        return (self.pt_cache_root / rel).with_suffix(".pt")
