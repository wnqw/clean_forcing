#!/usr/bin/env bash
# One-time environment setup for Clean Forcing (tested pattern; any A100/H100-class GPU, >=48GB).
set -e
conda create -y -n clean_forcing python=3.10
conda activate clean_forcing 2>/dev/null || source activate clean_forcing
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r self_forcing/requirements.txt
pip install pyiqa timm imageio[ffmpeg] imageio-ffmpeg decord einops omegaconf

# Base model weights (public): Wan2.1-T2V-1.3B -> self_forcing/wan_models/
pip install "huggingface_hub[cli]"
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir self_forcing/wan_models/Wan2.1-T2V-1.3B

# External Causal-Forcing base (public, Apache-2.0) — only needed for the lora_cf_* checkpoints:
# hf download zhuhz22/Causal-Forcing --include "chunkwise/ar_diffusion.pt" --local-dir cf_base

echo "Done. See RUN.md for the adapted-base checkpoint and demos."
