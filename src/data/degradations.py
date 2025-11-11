import io
import random

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur, to_pil_image, to_tensor
from PIL import Image


class DegradationConfig:
    def __init__(
        self,
        nearest_prob,
        box_prob,
        bilinear_prob,
        tiny_blur_prob,
        jpeg_prob,
        jpeg_quality_min,
        jpeg_quality_max,
        palette_jitter_prob=0.0,
        palette_jitter_std=0.0,
        channel_drop_prob=0.0,
        gaussian_noise_prob=0.0,
        gaussian_noise_std=0.0,
    ):
        self.nearest_prob = nearest_prob
        self.box_prob = box_prob
        self.bilinear_prob = bilinear_prob
        self.tiny_blur_prob = tiny_blur_prob
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_min = jpeg_quality_min
        self.jpeg_quality_max = jpeg_quality_max
        self.palette_jitter_prob = palette_jitter_prob
        self.palette_jitter_std = palette_jitter_std
        self.channel_drop_prob = channel_drop_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std = gaussian_noise_std
        self._validate()

    def _validate(self):
        for name in (
            "nearest_prob",
            "box_prob",
            "bilinear_prob",
            "tiny_blur_prob",
            "jpeg_prob",
            "palette_jitter_prob",
            "channel_drop_prob",
            "gaussian_noise_prob",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if self.palette_jitter_std < 0.0:
            raise ValueError(f"palette_jitter_std must be non-negative, got {self.palette_jitter_std}")
        if self.gaussian_noise_std < 0.0:
            raise ValueError(f"gaussian_noise_std must be non-negative, got {self.gaussian_noise_std}")
        if self.bilinear_prob + self.nearest_prob + self.box_prob > 1.0 + 1e-6:
            raise ValueError("sum of resample probabilities must be <= 1.0")
        if self.jpeg_quality_min > self.jpeg_quality_max:
            raise ValueError("jpeg_quality_min must be <= jpeg_quality_max")


def _pick_resample(config, r):
    if r < config.nearest_prob:
        return "nearest"
    r -= config.nearest_prob
    if r < config.box_prob:
        return "area"
    r -= config.box_prob
    if r < config.bilinear_prob:
        return "bilinear"
    # Fallback to bilinear when probability mass does not sum to 1 exactly.
    return "bilinear"


def _apply_jpeg_roundtrip(tensor, config, rng):
    if rng.random() >= config.jpeg_prob:
        return tensor
    quality = rng.randint(config.jpeg_quality_min, config.jpeg_quality_max)
    squeezed = tensor.squeeze(0).clamp(0.0, 1.0)
    channels = squeezed.shape[0]
    if channels == 1:
        pil = to_pil_image(squeezed)
        buffer = io.BytesIO()
        pil.convert("L").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        reloaded = Image.open(buffer).convert("L")
        return to_tensor(reloaded).unsqueeze(0)
    if channels == 3:
        pil = to_pil_image(squeezed)
        buffer = io.BytesIO()
        pil.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        reloaded = Image.open(buffer).convert("RGB")
        return to_tensor(reloaded).unsqueeze(0)
    if channels == 4:
        color = squeezed[:3]
        alpha = squeezed[3:].unsqueeze(0)
        pil = to_pil_image(color)
        buffer = io.BytesIO()
        pil.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        reloaded_color = to_tensor(Image.open(buffer).convert("RGB")).unsqueeze(0)
        return torch.cat([reloaded_color, alpha], dim=1)
    # Fallback for unexpected channel counts: skip JPEG roundtrip.
    return tensor


def _randn_like(tensor, rng):
    seed = rng.randint(0, 2**31 - 1)
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    return torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=generator)


def _apply_palette_jitter(tensor, config, rng):
    if rng.random() >= config.palette_jitter_prob:
        return tensor
    if config.palette_jitter_std <= 0.0:
        return tensor
    noise = _randn_like(tensor, rng) * config.palette_jitter_std
    return (tensor + noise).clamp(0.0, 1.0)


def _apply_channel_drop(tensor, config, rng):
    if rng.random() >= config.channel_drop_prob:
        return tensor
    channels = tensor.shape[1]
    if channels <= 1:
        return tensor
    clone = tensor.clone()
    drop = rng.randrange(0, channels)
    clone[:, drop : drop + 1] = 0.0
    return clone


def _apply_gaussian_noise(tensor, config, rng):
    if rng.random() >= config.gaussian_noise_prob:
        return tensor
    if config.gaussian_noise_std <= 0.0:
        return tensor
    noise = _randn_like(tensor, rng) * config.gaussian_noise_std
    return (tensor + noise).clamp(0.0, 1.0)


def downscale_with_profile(hr_image, scale, config, rng=None):
    if rng is None:
        rng = random
    tensor = to_tensor(hr_image).unsqueeze(0)
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    target_h = max(1, height // scale)
    target_w = max(1, width // scale)
    mode = _pick_resample(config, rng.random())
    if mode == "nearest":
        lr = F.interpolate(tensor, size=(target_h, target_w), mode="nearest")
    elif mode == "area":
        lr = F.interpolate(tensor, size=(target_h, target_w), mode="area")
    else:
        lr = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    if rng.random() < config.tiny_blur_prob:
        lr = gaussian_blur(lr, kernel_size=3, sigma=0.5)
    lr = _apply_palette_jitter(lr, config, rng)
    lr = _apply_channel_drop(lr, config, rng)
    lr = _apply_gaussian_noise(lr, config, rng)
    lr = _apply_jpeg_roundtrip(lr, config, rng)
    return lr.squeeze(0)
