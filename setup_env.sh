#!/bin/bash

# Usage: ./setup_env.sh 

# Install local editable packages

echo "[+] Installing 3D-Diffusion-Policy..."
cd 3D-Diffusion-Policy && pip install -e . && cd ..

echo "[+] Installing third_party/pytorch3d_simplified..."
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..


echo "[✓] Setup complete."
