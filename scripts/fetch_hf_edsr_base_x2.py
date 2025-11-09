import argparse
from pathlib import Path

import torch
from super_image import EdsrModel


def adapt_key(key):
    if key.startswith("module."):
        key = key[7:]
    if key.startswith("model."):
        key = key[6:]
    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod.") :]
    if key.startswith("sub_mean") or key.startswith("add_mean"):
        return None
    if key.startswith("head.0."):
        return "head." + key[len("head.0."):]
    if key.startswith("body.") and ".body." in key:
        parts = key.split(".")
        block_idx = parts[1]
        layer = parts[3]
        remainder = ".".join(parts[4:])
        if layer == "0":
            base = f"body.{block_idx}.conv1"
        elif layer == "2":
            base = f"body.{block_idx}.conv2"
        else:
            return None
        return f"{base}.{remainder}" if remainder else base
    if key.startswith("body.") and ".body." not in key:
        parts = key.split(".")
        if len(parts) >= 2 and parts[1] == "16":
            remainder = ".".join(parts[2:])
            return f"body_conv.{remainder}" if remainder else "body_conv"
    if key.startswith("body_conv"):
        return key
    if key.startswith("tail.0.0."):
        return "tail.0." + key[len("tail.0.0."):]
    if key.startswith("tail.1."):
        return "tail.2." + key[len("tail.1."):]
    if key.startswith("tail.") or key.startswith("head."):
        return key
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("loading eugenesiow/edsr-base scale 2 from Hugging Face")
    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)
    state = model.state_dict()
    mapped = {}
    for raw_key, tensor in state.items():
        new_key = adapt_key(raw_key)
        if new_key is None:
            continue
        mapped[new_key] = tensor.cpu()
    torch.save(mapped, output_path)
    print(f"saved mapped state dict ({len(mapped)} tensors) to {output_path}")


if __name__ == "__main__":
    main()
