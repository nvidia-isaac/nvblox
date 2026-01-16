#!/bin/bash

# On aarch64, the default pytorch wheels does not support CUDA. We therefore install custom versions.
# See https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 for more wheels.
set -e
if [ $SKIP_PYTORCH_INSTALL -eq 0 ]; then
    echo "Installing pytorch"
    . /opt/venv/bin/activate
    python3 -m pip install --ignore-installed --upgrade pip wheel setuptools

    # Install PyTorch/vision/torchaudio wheels built for JetPack 6 (JP6) with CUDA 12.6 (cu126),
    # which are compatible with cuDNN 9.x shipped in JetPack 6.1/6.2 (L4T r36.4.x).
    # Note: These wheels target aarch64 + Python 3.10.
    pip uninstall -y torch torchvision torchaudio || true
    pip install --no-cache-dir \
        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
else
    echo "Skipping pytorch installation"
fi
