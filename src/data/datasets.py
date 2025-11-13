import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import (
    load_image,
    choose_hr_crop,
    apply_augment,
    tensor_from_pil,
)
from .degradations import downscale_with_profile

_SOBEL_KERNEL_X = torch.tensor(
    [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)
_SOBEL_KERNEL_Y = _SOBEL_KERNEL_X.transpose(2, 3).contiguous()


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
        self.images = [load_image(path) for path in self.paths] if cache_images else None
        self._rngs = {}
        edge_cfg = edge_sampling or {}
        self.edge_sampling_enabled = bool(edge_cfg.get("enabled"))
        self.edge_candidates = max(1, int(edge_cfg.get("candidates", 1)))
        self.edge_prefer_top = float(edge_cfg.get("prefer_top_p", 0.5))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.images is not None:
            hr = self.images[index].copy()
        else:
            hr = load_image(self.paths[index])
        rng = self._resolve_rng()
        tile_aligned = rng.random() < self.crops_tile_aligned_prob
        hr = self._pick_patch(hr, tile_aligned, rng)
        if self.augment:
            hr = apply_augment(hr, self.flips, self.rotations, rng)
        lr_tensor = downscale_with_profile(hr, self.scale, self._active_degradation_config(), rng)
        hr_tensor = tensor_from_pil(hr)
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
