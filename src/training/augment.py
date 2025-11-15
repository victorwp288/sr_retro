import torch


_AUGMENT_SEED_MASK = 0x4F1BBCDC  # decorrelates augment RNG from upstream sample_seeds


def apply_gpu_augmentations(hr_batch, seed_values, allow_flips, allow_rotations):
    if not allow_flips and not allow_rotations:
        return hr_batch
    augmented = []
    batch_device = hr_batch.device
    generator_ready_device = batch_device if batch_device.type in {"cuda", "cpu"} else torch.device("cpu")
    for tensor, seed in zip(hr_batch, seed_values):
        patch = tensor
        generator = torch.Generator(device=generator_ready_device)
        generator.manual_seed((int(seed) ^ _AUGMENT_SEED_MASK) & 0xFFFFFFFF)
        rand_vals = torch.rand(3, device=generator_ready_device, generator=generator)
        if generator_ready_device != batch_device:
            rand_vals = rand_vals.to(batch_device)
        if allow_flips and rand_vals[0].item() < 0.5:
            patch = torch.flip(patch, dims=(-1,))
        if allow_flips and rand_vals[1].item() < 0.5:
            patch = torch.flip(patch, dims=(-2,))
        if allow_rotations and rand_vals[2].item() < 0.5:
            patch = torch.flip(patch, dims=(-1, -2))
        augmented.append(patch)
    return torch.stack(augmented, dim=0)
