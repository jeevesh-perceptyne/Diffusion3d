#!/bin/bash

# Usage: ./setup_env.sh [env_name]
# If no env_name is provided, installs everything in the current environment

ENV_NAME=$1

if [ ! -z "$ENV_NAME" ]; then
  echo "[+] Creating conda environment: $ENV_NAME"
  conda create -y -n $ENV_NAME python=3.10
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate $ENV_NAME
else
  echo "[!] No environment name provided. Installing in current environment."
fi

# Install PyTorch for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install local editable packages
echo "[+] Installing 3D-Diffusion-Policy..."
cd 3D-Diffusion-Policy && pip install -e . && cd ..

echo "[+] Installing third_party/pytorch3d_simplified..."
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..

# Install other Python dependencies
pip install \
  zarr==2.12.0 \
  wandb \
  ipdb \
  gpustat \
  dm_control \
  omegaconf \
  hydra-core==1.2.0 \
  dill==0.3.5.1 \
  einops==0.4.1 \
  diffusers==0.11.1 \
  numba==0.56.4 \
  moviepy \
  imageio \
  av \
  matplotlib \
  termcolor \
  boto3

echo "[✓] Setup complete."
