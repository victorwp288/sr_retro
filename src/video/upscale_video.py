import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.models import EDSR
from src.video.temporal import mean3, flow_blend


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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stabilizer", choices=["none", "mean3", "flow"], default="flow")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=["video", "image"], default="video")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    model_cfg = config["model"]
    device = pick_device(args.device)
    print(f"running on {device}")

    model = EDSR(
        scale=model_cfg["scale"],
        n_feats=model_cfg["n_feats"],
        n_resblocks=model_cfg["n_resblocks"],
        res_scale=model_cfg["res_scale"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    model.eval()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "image":
        image_bgr = cv2.imread(input_path.as_posix(), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"could not open image {input_path}")
        lr_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        lr_norm = lr_rgb.astype(np.float32) / 255.0
        lr_tensor = torch.from_numpy(lr_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp(0.0, 1.0)
        sr_frame = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_bgr = cv2.cvtColor((sr_frame * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path.as_posix(), sr_bgr)
        print(f"saved upscaled image to {output_path}")
        return

    cap = cv2.VideoCapture(input_path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = model_cfg["scale"]
    out_size = (width * scale, height * scale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, out_size)

    history = []
    prev_lr = None
    prev_sr = None

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            lr_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lr_norm = lr_rgb.astype(np.float32) / 255.0
            lr_tensor = torch.from_numpy(lr_norm).permute(2, 0, 1).unsqueeze(0).to(device)
            sr_tensor = model(lr_tensor).clamp(0.0, 1.0)
            sr_frame = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

            if args.stabilizer == "mean3":
                history.append(sr_frame)
                if len(history) > 3:
                    history.pop(0)
                sr_frame = mean3(history)
            elif args.stabilizer == "flow" and prev_lr is not None and prev_sr is not None:
                sr_frame = flow_blend(prev_lr, lr_norm, prev_sr, sr_frame, flow_strength=0.35)

            prev_lr = lr_norm
            prev_sr = sr_frame.copy()

            sr_bgr = cv2.cvtColor((sr_frame * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(sr_bgr)

    cap.release()
    writer.release()
    print(f"saved upscaled video to {output_path}")


if __name__ == "__main__":
    main()
