import torch
from torch.utils.data import Dataset

from .utils import (
    load_image,
    choose_hr_crop,
    apply_augment,
    tensor_from_pil,
)
from .degradations import downscale_with_profile, DegradationConfig


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
    ):
        if not paths:
            raise RuntimeError("dataset received empty path list")
        self.paths = paths
        self.scale = scale
        self.patch_size_hr = patch_size_hr
        self.patch_size_ref = patch_size_ref
        self.degradation_config = degradation_config
        self.crops_tile_aligned_prob = crops_tile_aligned_prob
        self.flips = flips
        self.rotations = rotations
        self.augment = augment
        self.cache_images = cache_images
        self.images = [load_image(path) for path in self.paths] if cache_images else None
        self._rngs = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.images is not None:
            hr = self.images[index].copy()
        else:
            hr = load_image(self.paths[index])
        rng = self._resolve_rng()
        tile_aligned = rng.random() < self.crops_tile_aligned_prob
        hr = choose_hr_crop(hr, self._current_patch_size(), self.scale, tile_aligned, rng)
        if self.augment:
            hr = apply_augment(hr, self.flips, self.rotations, rng)
        lr_tensor = downscale_with_profile(hr, self.scale, self.degradation_config, rng)
        hr_tensor = tensor_from_pil(hr)
        return lr_tensor, hr_tensor

    def _current_patch_size(self):
        if self.patch_size_ref is not None:
            return int(self.patch_size_ref.value)
        return self.patch_size_hr

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
