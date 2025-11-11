# Retro Pixel Super Resolution

Retro Pixel Super Resolution is a small, practical training stack for super‑resolving **pixel art and retro game sprites/tiles**. It fine‑tunes an EDSR‑style backbone on synthetic low‑resolution inputs produced on‑the‑fly with a blend of downscale kernels and capture artifacts. The recipe is:

- Warm‑start from generic EDSR weights.
- Train **x2** first, then reuse the learned trunk and train **x4**.
- Use tile‑aligned crops and light flips/rotations; degradations are configured in YAML.
- Validate with average L1/PSNR/SSIM; selection is **PSNR** by default.
- Run on CUDA when available, otherwise MPS (Apple Silicon) or CPU.

## Quickstart

1. Create a virtual environment and install packages.
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```
2. Fetch warm-start weights from Hugging Face and save a state dict.
   ```bash
   python scripts/fetch_hf_edsr_base_x2.py --output pretrained/edsr_base_x2.pt
   ```
3. Train ×2 tail → ×2 trunk → ×4 tail → ×4 trunk (see “Recommended Workflow” below for the exact commands). Always invoke the CLIs as modules so `src` stays on the import path.

## Recommended Workflow

The current recipe adds dynamic patch schedules, Charbonnier/L1/gradient losses, richer degradations, and automatic early stopping. Run the stages below in order (resume with `--resume_from …/last.pth` only if a job stops mid-way):

```bash
# Optional: confirm the dataloader settings on your machine
python -m scripts.benchmark_loader --config configs/edsr_64x16_x2_tail.yaml --steps 400 --warmup 100

# Stage 1: ×2 tail-only fine-tune (warm-start from HF weights)
python -m src.train --config configs/edsr_64x16_x2_tail.yaml

# Stage 2: ×2 full trunk
python -m src.train --config configs/edsr_64x16_x2_trunk.yaml

# Stage 3: ×4 tail-only (loads the finished ×2 trunk)
python -m src.train --config configs/edsr_64x16_x4_tail.yaml

# Stage 4: ×4 full trunk (loads the ×4 tail checkpoint)
python -m src.train --config configs/edsr_64x16_x4_trunk.yaml

# Mega evaluation: PSNR/SSIM/L1/LPIPS + example strips for every stage
python -m src.eval --config configs/edsr_64x16_x4_trunk.yaml \
  --models tail_x2=runs/edsr_x2/best.pth \
            trunk_x2=runs/edsr_x2_trunk/best.pth \
            twopass_x2=twopass:runs/edsr_x2_trunk/best.pth \
            tail_x4=runs/edsr_x4/best.pth \
            trunk_x4=runs/edsr_x4_trunk/best.pth \
  --out eval_out_\$(date +%Y%m%d)_mega
```

**Resume training** (continues optimizer + AMP scaler and skips warm‑up if resuming past it):
```bash
# Resume ×2 tail or trunk
python -m src.train --config configs/edsr_64x16_x2_tail.yaml --resume_from runs/edsr_x2/last.pth
python -m src.train --config configs/edsr_64x16_x2_trunk.yaml --resume_from runs/edsr_x2_trunk/last.pth

# Resume ×4 tail or trunk
python -m src.train --config configs/edsr_64x16_x4_tail.yaml --resume_from runs/edsr_x4/last.pth
python -m src.train --config configs/edsr_64x16_x4_trunk.yaml --resume_from runs/edsr_x4_trunk/last.pth
```
Checkpoints are written to `runs/edsr_x{2,4}/` as:
- `best.pth` — best validation PSNR so far (selection metric is PSNR).
- `last.pth` — rolling checkpoint written at each validation.

The dataloader walks `data`, performs deterministic train/val/test auto-splits, and downscales high-resolution tiles and sprites on the fly using a blend of kernels and capture artifacts to produce training low-resolution inputs. When editing `degradations` in a config, only `nearest_prob` and `box_prob` are specified—the bilinear kernel automatically receives the remaining probability mass.

## Evaluate

`src.eval` now emits PSNR, SSIM, L1, and LPIPS (unless you pass `--no_lpips`) for every checkpoint you list via `--models`. Use the “mega” command above to compare all stages at once, or trim the model list for focused comparisons. Flags such as `--crop`, `--y`, and `--device` still work; LPIPS automatically falls back to RGB tensors even if you evaluate PSNR/SSIM on the Y channel.

## Configuration

Each file in `configs/` follows this shape (field names used by the code):

```yaml
seed: 42

model:
  scale: 2            # or 4
  n_feats: 64         # "64x16" -> 64 features, 16 residual blocks
  n_resblocks: 16
  res_scale: 1.0

data:
  root: data                  # or use explicit {train, val} split files
  patch_size_hr: 64           # HR crop; LR will be patch_size_hr / scale
  crops_tile_aligned_prob: 0.7
  flips: true
  rotations: true
  train_workers: 4
  val_workers: 2
  cache_images: true

degradations:
  nearest_prob: 0.4
  box_prob: 0.3               # bilinear gets (1 - nearest - box)

training:
  steps: 200000
  batch_size: 16
  tail_only: true             # fine‑tune the upscaling tail first
  amp: true                   # automatic mixed precision on CUDA
  compile: true               # only effective on CUDA
  val_every: 1000
  warmup_steps: 2000
  warmup_start_factor: 1e-3
  # pretrained_path: pretrained/edsr_base_x2.pt
  # baseline_path:   pretrained/edsr_base_x2.pt

optimizer:
  type: adamw                 # or "adam"
  lr_trunk: 1.0e-4            # used if tail_only: false
  lr_tail:  2.0e-4
  weight_decay: 0.0

eval:
  crop_border: 0              # typically set to scale for SR benchmarks
  y_channel: false            # SR-style PSNR/SSIM on Y if true
```

> Tip: The provided `configs/edsr_64x16_x2.yaml`, `configs/edsr_64x16_x4.yaml`, and `configs/edsr_64x16_x2_trunk.yaml` follow this schema. Train x2 first, then x4; use the trunk variant once you're ready to fine-tune the full backbone.

## Checkpoints & Baselines

- **Saving:** `last.pth` is written at each validation; `best.pth` is saved when PSNR improves.
- **Resume:** `--resume_from runs/edsr_x{2,4}/last.pth` restores model, optimizer, and AMP scaler states. Warm‑up LR is **not** re‑applied when resuming beyond the warm‑up window.
- **Baseline eval (optional):** If you set `baseline_path` in the config and the shapes match, the script will log baseline L1/PSNR/SSIM on the validation split before training. Useful to sanity‑check the warm‑start weights.

## Training Tips

- **Selection metric:** PSNR (fixed in the script).
- **Optimizer:** AdamW by default (`optimizer.type`), weight decay defaults to 0.0. Set `type: adam` to match older EDSR settings.
- **Learning rate:** Linear warm‑up via `warmup_steps`/`warmup_start_factor`. No decay schedule is applied by default.
- **Precision & speed:** On CUDA we enable high matmul precision (TF32 on Ampere‑class GPUs) and optional `torch.compile`. On Apple Silicon, MPS is supported; on other systems it falls back to CPU.
- **Crops:** Training uses tile‑aligned crops with light flips/rotations. Validation/eval can optionally crop borders (`eval.crop_border`) and compute metrics on the Y channel (`eval.y_channel`).

## Implementation Notes & Updates

- **Device selection:** The code auto‑picks **CUDA → MPS → CPU**. If you pass a preferred device, availability is validated (e.g., requesting `"cuda"` without a GPU raises a clear error).
- **SSIM safety:** Small sprites are common. SSIM uses a safe kernel on tiny inputs; evaluation also validates `crop_border` so slicing never produces negative extents.
- **Warm‑start key adaptation:** Pretrained state dict keys are adapted (e.g., `module.*`, `head.0.*`, block naming), and mean‑shift modules are ignored.
- **Tail‑only fine‑tuning:** When `training.tail_only: true`, only the upsampling tail is trainable and gets its own LR. Full‑model training is supported by turning it off.
- **Per‑sample metrics:** PSNR is computed per sample and averaged, matching common SR reporting.
- **Rolling + best checkpoints:** `last.pth` and `best.pth` are maintained separately.

## Troubleshooting

- **"Requested CUDA but it isn't available":** Either remove the preferred device flag or install a CUDA build of PyTorch. The script automatically falls back to MPS (Apple Silicon) or CPU when not forced.
- **"Invalid crop_border for tensor of shape …":** Ensure `--crop_border ≤ min(H, W) // 2` for your validation images; small sprites can be very tiny.
- **SSIM on tiny tiles:** No change needed—SSIM is guarded internally. If you still need to disable SSIM, omit the `--y_channel` and `--lpips` flags and rely on L1/PSNR.
- **Resume resets warm‑up:** Fixed—resuming preserves optimizer LRs and skips warm‑up past the saved step.

## Export ONNX

```bash
python -m src.export_onnx --config configs/edsr_64x16_x4.yaml --checkpoint runs/edsr_x4/best.pth --output onnx/edsr_x4.onnx
```

## Quick Pilot

```bash
# Run a longer pilot to finish warm-up and see PSNR/SSIM trend
python -m src.train --config configs/edsr_64x16_x2.yaml --pilot
```

## Hyperparameter Sweep (optional)

```bash
# Runs a compact grid and reports best validation L1
python -m scripts.run_sweep --config configs/edsr_64x16_x2.yaml --steps 10000 --top 5

# Loader benchmark to tune workers/prefetch
python -m scripts.benchmark_loader --config configs/edsr_64x16_x2.yaml --steps 200 --warmup 50
```

## Eval

The command below uses the provided `configs/edsr_64x16_x2_trunk.yaml`, which trains/evaluates the full trunk after the tail-only phase.
```bash
# Full eval
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m src.eval \
  --config configs/edsr_64x16_x2_trunk.yaml \
  --out eval_out \
  --dump 0 \
  --models \
  base_x2=pretrained/edsr_base_x2.pt \
  tail_x2=runs/edsr_x2/best.pth \
  trunk_x2=runs/edsr_x2_trunk/best.pth \
  base_x4_twopass=twopass:pretrained/edsr_base_x2.pt \
  tail_x4=runs/edsr_x4/best.pth \
  trunk_x4=runs/edsr_x4_trunk/best.pth
```

## Upscale Video or Image

```bash
python -m src.video.upscale_video --config configs/edsr_64x16_x4.yaml --checkpoint runs/edsr_x4/best.pth --input path/to/input.mp4 --output outputs/upscaled.mp4 --stabilizer flow
python -m src.video.upscale_video --config configs/edsr_64x16_x4.yaml --checkpoint runs/edsr_x4/best.pth --mode image --input data/example.png --output outputs/example_x4.png
```

## Glossary

- **PSNR**: Peak Signal-to-Noise Ratio, higher is better and tracks fidelity in dB.
- **SSIM**: Structural Similarity Index, measures structural overlap between images.
- **LPIPS**: Learned Perceptual Image Patch Similarity, lower is better for perceptual distance.
- **Residual scaling**: Shrinks residual block outputs before adding them back to keep training stable.
- **PixelShuffle**: Rearranges channel data into spatial resolution for efficient upscaling.
