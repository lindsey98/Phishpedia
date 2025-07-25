#!/bin/bash

set -euo pipefail  # Safer bash behavior
IFS=$'\n\t'

# Install Detectron2
pixi run pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

# Set up model directory
FILEDIR="$(pwd)"
MODELS_DIR="$FILEDIR/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Download model files
pixi run gdown --id "1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH" -O "rcnn_bet365.pth"
pixi run gdown --id "1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW" -O "faster_rcnn.yaml"
pixi run gdown --id "1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS" -O "resnetv2_rgb_new.pth.tar"
pixi run gdown --id "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I" -O "expand_targetlist.zip"
pixi run gdown --id "1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1" -O "domain_map.pkl"

# Extract and flatten expand_targetlist
echo "Extracting expand_targetlist.zip..."
unzip -o expand_targetlist.zip -d expand_targetlist

cd expand_targetlist || error_exit "Extraction directory missing."

if [ -d "expand_targetlist" ]; then
  echo "Flattening nested expand_targetlist/ directory..."
  mv expand_targetlist/* .
  rm -r expand_targetlist
fi

echo "Model setup and extraction complete."
