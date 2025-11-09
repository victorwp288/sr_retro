import argparse
from pathlib import Path

import torch
import yaml

from src.models import EDSR


def pick_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    model_cfg = config["model"]
    device = pick_device(args.device)
    print(f"exporting with {device}")

    model = EDSR(
        scale=model_cfg["scale"],
        n_feats=model_cfg["n_feats"],
        n_resblocks=model_cfg["n_resblocks"],
        res_scale=model_cfg["res_scale"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    model.eval()

    scale = model_cfg["scale"]
    patch_hr = args.patch or config["data"].get("patch_size_hr") or 128
    patch_lr = max(8, patch_hr // scale)
    dummy = torch.randn(1, 3, patch_lr, patch_lr, dtype=torch.float32, device=device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path.as_posix(),
        input_names=["lr"],
        output_names=["sr"],
        opset_version=17,
        dynamic_axes={"lr": {2: "h", 3: "w"}, "sr": {2: "H", 3: "W"}},
    )
    print(f"exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
