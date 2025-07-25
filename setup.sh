#!/bin/bash

set -euo pipefail  # Safer bash behavior
IFS=$'\n\t'

# Logging function with timestamp
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Function to display and log errors, then exit
error_exit() {
  log "ERROR: $1" >&2
  echo "$(date): $1" >> error.log
  exit 1
}

# 4. Retryable download function
RETRY_COUNT=3
download_with_retry() {
  local file_id="$1"
  local file_name="$2"
  local count=0

  until [ $count -ge $RETRY_COUNT ]; do
    log "Downloading $file_name (Attempt $((count + 1))/$RETRY_COUNT)..."
    pixi run gdown --id "$file_id" -O "$file_name" && return
    count=$((count + 1))
    sleep 2
  done

  error_exit "Failed to download $file_name after $RETRY_COUNT attempts."
}


# Set up model directory
FILEDIR="$(pwd)"
MODELS_DIR="$FILEDIR/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# ALL Model files
declare -A MODEL_FILES=(
  ["rcnn_bet365.pth"]="1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH"
  ["faster_rcnn.yaml"]="1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW"
  ["resnetv2_rgb_new.pth.tar"]="1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS"
  ["expand_targetlist.zip"]="1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I"
  ["domain_map.pkl"]="1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1"
)

# Install Detectron2
pixi run pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

# Download model files
for file_name in "${!MODEL_FILES[@]}"; do
  file_id="${MODEL_FILES[$file_name]}"
  if [ -f "$file_name" ]; then
    log "$file_name already exists. Skipping."
  else
    download_with_retry "$file_id" "$file_name"
  fi
done

# 6. Extract and flatten expand_targetlist
log "Extracting expand_targetlist.zip..."
unzip -o expand_targetlist.zip -d expand_targetlist

cd expand_targetlist || error_exit "Extraction directory missing."

if [ -d "expand_targetlist" ]; then
  log "Flattening nested expand_targetlist/ directory..."
  shopt -s dotglob
  mv expand_targetlist/* .
  shopt -u dotglob
  rmdir expand_targetlist
fi

cd "$MODELS_DIR"

log "Model setup and extraction complete."
