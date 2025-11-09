#!/bin/bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"

# Upgrade pip first
"${python_bin}" -m pip install --upgrade pip

# Ensure no conflicting OpenCV wheels linger.
"${python_bin}" -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless >/dev/null 2>&1 || true

# Install PyTorch 2.4.0 + torchvision 0.19.0. Prefer CUDA 12 wheels when available,
# otherwise fall back to CPU-only wheels.
if ! "${python_bin}" -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 ; then
    echo "Falling back to CPU-only PyTorch wheels"
    "${python_bin}" -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Core project dependencies
"${python_bin}" -m pip install Pillow numpy pyyaml tqdm scikit-image lpips torchmetrics matplotlib super-image
"${python_bin}" -m pip install --no-cache-dir opencv-python-headless

echo "Environment setup complete."
