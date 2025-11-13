# Retro Pixel Super Resolution

Retro Pixel Super Resolution is a practical SR stack for **pixel art and retro game sprites/tiles**. The new “Balanced Pixel‑SR” recipe expands the backbone to 64×32 blocks and layers in stability features that target razor‑sharp PSNR while suppressing halos:

- Warm‑start from the generic ×2 EDSR weights, then grow the network depth (64×32) while adding a residual output head outside the model so pretrained checkpoints remain compatible.
- Follow a fixed stage flow — ×2 tail → ×2 trunk → ×4 tail → ×4 trunk — with EMA + SWA tracking, cosine LR, gradient clipping, and 4‑step patch schedules (64→96→128→160 for ×2).
- Blend curriculum degradations (nearest/area/bilinear ratios, JPEG/noise ramps) with edge‑aware sampling (4 Sobel‑ranked crops per batch) so fine outlines stay crisp.
- Distill every heavy stage from the best ×2 trunk (two‑pass for ×4, AMP + channels_last), while late‑phase LPIPS and TTA (8‑way) polish the outputs.
- Validate on PSNR/SSIM/L1 + Gradient‑L1 (Y channel) with optional LPIPS, and run anywhere: CUDA → MPS → CPU.

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

The Balanced Pixel‑SR stack now layers EMA/SWA, cosine warm‑ups, distillation, curriculum degradations, and edge‑aware sampling on top of the 64×32 backbone. Run the four stages below in order (resume with `--resume_from …/last.pth` only if a job stops mid‑way):

```bash
# Optional: benchmark dataloaders on your box
python -m scripts.benchmark_loader --config configs/edsr_64x32_x2_tail.yaml --steps 400 --warmup 100

# Stage 1 — ×2 tail (adapts the upsampler, trunk frozen, warm-starts from HF weights)
python -m src.train --config configs/edsr_64x32_x2_tail.yaml

# Stage 2 — ×2 trunk (full network, EMA/SWA on, cosine LR, gradient clipping)
python -m src.train --config configs/edsr_64x32_x2_trunk.yaml

# Stage 3 — ×4 tail (loads the ×2 trunk, trains the ×4 upsampler only)
python -m src.train --config configs/edsr_64x32_x4_tail.yaml

# Stage 4 — ×4 trunk (full network at ×4, teacher = two-pass ×2 trunk)
python -m src.train --config configs/edsr_64x32_x4_trunk.yaml

# Mega evaluation: PSNR/SSIM/L1/Gradient-L1/LPIPS (+ example strips, 8-way TTA)
python -m src.eval --config configs/edsr_64x32_x4_trunk.yaml \
  --models \
    base_twopass=twopass:pretrained/edsr_base_x2.pt \
    ema_x2_tail=ema:runs/pxsr_x2_tail/best.pth \
    ema_x2_trunk=ema:runs/pxsr_x2_trunk/best.pth \
    swa_x4_tail=swa:runs/pxsr_x4_tail/final_swa.pth \
    ema_x4_trunk=ema:runs/pxsr_x4_trunk/best.pth \
    twopass_x4=twopass:runs/pxsr_x2_trunk/best.pth \
  --out eval_out_$(date +%Y%m%d)_balanced
```

**Resume training** (continues optimizer + AMP scaler and skips warm‑up if resuming past it):
```bash
# Resume ×2 tail or trunk
python -m src.train --config configs/edsr_64x32_x2_tail.yaml --resume_from runs/pxsr_x2_tail/last.pth
python -m src.train --config configs/edsr_64x32_x2_trunk.yaml --resume_from runs/pxsr_x2_trunk/last.pth

# Resume ×4 tail or trunk
python -m src.train --config configs/edsr_64x32_x4_tail.yaml --resume_from runs/pxsr_x4_tail/last.pth
python -m src.train --config configs/edsr_64x32_x4_trunk.yaml --resume_from runs/pxsr_x4_trunk/last.pth
```
Checkpoints are written to `runs/pxsr_x{2,4}_*/` as:
- `best.pth` — best validation PSNR so far (selection metric is PSNR).
- `last.pth` — rolling checkpoint written at each validation.

The dataloader walks `data`, performs deterministic train/val/test auto-splits, and downscales high-resolution tiles and sprites on the fly using a blend of kernels and capture artifacts to produce training low-resolution inputs. When editing `degradations` in a config, only `nearest_prob` and `box_prob` are specified—the bilinear kernel automatically receives the remaining probability mass.

## Evaluate

`src.eval` now emits PSNR, SSIM, L1, Gradient‑L1, and LPIPS (unless you pass `--no_lpips`) for every checkpoint listed via `--models`. Prefix a model path with `ema:` or `swa:` to load those states, or `twopass:` to run a ×2 model twice for ×4 baselines. Use the “mega” command above to compare all stages at once, or trim the model list for focused comparisons. Flags such as `--crop`, `--y`, and `--device` still work; LPIPS automatically falls back to RGB tensors even if you evaluate PSNR/SSIM on the Y channel.

## Configuration

Each file in `configs/` follows the new schema below (field names used by the code):

```yaml
seed: 42

model:
  scale: 2                  # or 4
  n_feats: 64
  n_resblocks: 32
  res_scale: 0.1
  residual_output: true     # add nearest-upsampled LR outside the model

training:
  steps: 400000
  batch_size: 40            # 20 for ×4
  tail_only: true           # set false for trunk stages
  amp: true
  compile: true
  val_every: 1000
  warmup_steps: 8000
  warmup_start_factor: 0.001
  patch_mini_warmup_steps: 500
  scheduler:
    type: cosine
    min_lr_factor: 1.0e-2
    final_min_lr_factor: 5.0e-3
    final_phase_start: 0.9
  ema: { enabled: true, decay: 0.99995, update_every: 1 }
  swa: { enabled: true, start_frac: 0.8, update_every: 5000 }
  grad_clip: { enabled: true, max_norm: 1.0 }

optimizer:
  type: adam
  lr_trunk: 5.0e-5          # set to 0 when tail_only: true
  lr_tail:  2.0e-4
  weight_decay: 0.0

loss:
  y_only: true
  charbonnier_eps: 0.001
  schedule:
    - { step: 0,   charbonnier: 1.0, rgb_l1: 0.05, grad: 0.10, lpips: 0.00, lpips_net: alex }
    - { step: 0.8T, charbonnier: 1.0, rgb_l1: 0.05, grad: 0.10, lpips: 0.02, lpips_net: alex }

distillation:
  enabled: true
  teacher_path: runs/pxsr_x2_trunk/best.pth   # x4 stages also set teacher_twopass: true
  teacher_twopass: false
  start_frac: 0.6
  final_frac: 0.9
  weight_max: 0.10
  weight_final: 0.12
  sample_frac: 0.5

degradations:
  nearest_prob: 0.65
  box_prob: 0.20
  bilinear_prob: 0.15
  tiny_blur_prob: 0.05
  jpeg_prob: 0.20
  jpeg_quality_min: 75
  jpeg_quality_max: 95
  palette_jitter_prob: 0.25
  palette_jitter_std: 0.02
  channel_drop_prob: 0.05
  gaussian_noise_prob: 0.10
  gaussian_noise_std: 0.005

data:
  root: data
  patch_size_hr: 128
  patch_schedule:
    - { step: 0, size: 64 }
    - { step: 133333, size: 96 }
    - { step: 266666, size: 128 }
    # + { step: 360000, size: 160 } for ×2 stages
  crops_tile_aligned_prob: 1.0
  flips: true
  rotations: true
  train_workers: 12
  val_workers: 4
  edge_sampling: { enabled: true, candidates: 4, prefer_top_p: 0.6 }
  degrad_schedule:
    - { step: 0, jpeg_prob: 0.10, gaussian_noise_prob: 0.05 }
    - { step: 0.6T, jpeg_prob: 0.20, gaussian_noise_prob: 0.10 }

eval:
  crop_border: 2            # 4 for ×4 stages
  y_channel: true
  tta: true
  dump: 4
```

> Tip: the four configs in `configs/edsr_64x32_*.yaml` already encode the stage order, pretrained paths, and teacher checkpoints. Train ×2 tail → ×2 trunk → ×4 tail → ×4 trunk, running `src.eval` after any stage to compare EMA vs SWA outputs.

## Checkpoints & Baselines

- **Saving:** `last.pth` is written at each validation; `best.pth` is saved when PSNR improves.
- **Resume:** `--resume_from runs/pxsr_x{2,4}_*/last.pth` restores model, optimizer, and AMP scaler states. Warm‑up LR is **not** re‑applied when resuming beyond the warm‑up window.
- **Baseline eval (optional):** If you set `baseline_path` in the config and the shapes match, the script will log baseline L1/PSNR/SSIM on the validation split before training. Useful to sanity‑check the warm‑start weights.
- **SWA artifacts:** Every stage exports `final_swa.pth` (pure EMA‑SWA weights plus metrics). When SWA beats EMA at the end, `best.pth` is “promoted” to the SWA model and includes `swa_promoted: true` (optimizer state is omitted because it no longer matches the averaged weights). Use `last.pth` if you need a resume‑ready optimizer snapshot.

## Training Tips

- **Selection metric:** PSNR (fixed in the script).
- **Optimizer:** AdamW by default (`optimizer.type`), weight decay defaults to 0.0. Set `type: adam` to match older EDSR settings.
- **Learning rate:** Linear warm‑up via `warmup_steps`/`warmup_start_factor`. No decay schedule is applied by default.
- **Precision & speed:** On CUDA we enable high matmul precision (TF32 on Ampere‑class GPUs) and optional `torch.compile`. On Apple Silicon, MPS is supported; on other systems it falls back to CPU.
- **Crops:** Training uses tile‑aligned crops with light flips/rotations. Validation/eval can optionally crop borders (`eval.crop_border`) and compute metrics on the Y channel (`eval.y_channel`).
- **Determinism:** `set_seed()` now seeds Python, NumPy, and PyTorch RNGs; distillation dropout uses `torch.rand`, so as long as you keep the same seed and config, runs are reproducible.

## Implementation Notes & Updates

- **Device selection:** The code auto‑picks **CUDA → MPS → CPU**. If you pass a preferred device, availability is validated (e.g., requesting `"cuda"` without a GPU raises a clear error).
- **SSIM safety:** Small sprites are common. SSIM uses a safe kernel on tiny inputs; evaluation also validates `crop_border` so slicing never produces negative extents.
- **Warm‑start key adaptation:** Pretrained state dict keys are adapted (e.g., `module.*`, `head.0.*`, block naming), and mean‑shift modules are ignored.
- **Tail‑only fine‑tuning:** When `training.tail_only: true`, only the upsampling tail is trainable and gets its own LR. Full‑model training is supported by turning it off.
- **Per‑sample metrics:** PSNR is computed per sample and averaged, matching common SR reporting.
- **Rolling + best checkpoints:** `last.pth` and `best.pth` are maintained separately.
- **Full-range Y luminance:** When `loss.y_only: true`, the pipeline converts RGB to Y using the full-range BT.601 weights (`Y = 0.299 R + 0.587 G + 0.114 B`), matching our [0,1] normalized tensors.


## Quick Pilot

```bash
# Run a longer pilot to finish warm-up and see PSNR/SSIM trend
python -m src.train --config configs/edsr_64x32_x2_tail.yaml --pilot
```

## Hyperparameter Sweep (optional)

```bash
# Runs a compact grid and reports best validation L1
python -m scripts.run_sweep --config configs/edsr_64x32_x2_tail.yaml --steps 10000 --top 5

# Loader benchmark to tune workers/prefetch
python -m scripts.benchmark_loader --config configs/edsr_64x32_x2_tail.yaml --steps 200 --warmup 50
```

## Eval

The command below uses `configs/edsr_64x32_x2_trunk.yaml`, which evaluates the ×2 trunk after the tail-only phase (swap configs to evaluate ×4).
```bash
# Full eval
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m src.eval \
  --config configs/edsr_64x32_x2_trunk.yaml \
  --out eval_out \
  --dump 0 \
  --models \
  base_x2=pretrained/edsr_base_x2.pt \
  ema_x2_tail=ema:runs/pxsr_x2_tail/best.pth \
  ema_x2_trunk=ema:runs/pxsr_x2_trunk/best.pth \
  base_x4_twopass=twopass:pretrained/edsr_base_x2.pt \
  ema_x4_tail=ema:runs/pxsr_x4_tail/best.pth \
  ema_x4_trunk=ema:runs/pxsr_x4_trunk/best.pth
```

## Frame Extraction Utility

Need to bulk up the dataset from longplay videos or trailers? Use the helper script to sample frames at a fixed stride:

```bash
python -m scripts.extract_frames \
  --video downloads/retro_longplay.mp4 \
  --out data/frames/longplay_01 \
  --every 8 \
  --start 600 \
  --max_frames 3000 \
  --prefix longplay
```

- `--every` keeps one out of every N frames so adjacent duplicates don’t dominate.
- `--start` skips opening credits/menus, while `--max_frames` limits how many PNGs are saved per source.
- Outputs are PNGs named `<prefix>_000000.png`, ready to drop into your `data` directory or a custom split file.

## Upscale Video or Image

```bash
python -m src.video.upscale_video --config configs/edsr_64x32_x4_trunk.yaml --checkpoint runs/pxsr_x4_trunk/best.pth --input path/to/input.mp4 --output outputs/upscaled.mp4 --stabilizer flow
python -m src.video.upscale_video --config configs/edsr_64x32_x4_trunk.yaml --checkpoint runs/pxsr_x4_trunk/best.pth --mode image --input data/example.png --output outputs/example_x4.png
```

## Glossary

- **PSNR**: Peak Signal-to-Noise Ratio, higher is better and tracks fidelity in dB.
- **SSIM**: Structural Similarity Index, measures structural overlap between images.
- **LPIPS**: Learned Perceptual Image Patch Similarity, lower is better for perceptual distance.
- **Residual scaling**: Shrinks residual block outputs before adding them back to keep training stable.
- **PixelShuffle**: Rearranges channel data into spatial resolution for efficient upscaling.
