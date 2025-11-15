import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract every Nth frame from a video into PNGs.")
    parser.add_argument("--video", required=True, help="Path to the source video file.")
    parser.add_argument("--out", required=True, help="Directory where frames will be written.")
    parser.add_argument("--every", type=int, default=12, help="Stride: keep 1 frame out of N (default: 12).")
    parser.add_argument("--start", type=int, default=0, help="Skip this many frames before sampling.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on number of saved frames.")
    parser.add_argument("--prefix", default="frame", help="Filename prefix for saved PNGs.")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video_path}")

    keep_every = max(1, args.every)
    start = max(0, args.start)
    saved = 0
    total = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if total < start:
            total += 1
            continue
        if (total - start) % keep_every == 0:
            filename = out_dir / f"{args.prefix}_{saved:06d}.png"
            cv2.imwrite(filename.as_posix(), frame_bgr)
            saved += 1
            if args.max_frames is not None and saved >= args.max_frames:
                break
        total += 1

    cap.release()
    print(f"saved {saved} frames (sampled 1 out of every {keep_every} starting at frame {start})")


if __name__ == "__main__":
    main()
