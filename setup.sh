#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to display error messages and exit
error_exit() {
  echo "$1" >&2
  exit 1
}

# 1. Set ENV_NAME with default value "phishpedia" if not already set
ENV_NAME="${ENV_NAME:-phishpedia}"

# 2. Check if ENV_NAME is set (it always will be now, but kept for flexibility)
if [ -z "$ENV_NAME" ]; then
  error_exit "ENV_NAME is not set. Please set the environment name and try again."
fi

# 3. Set retry count for downloads
RETRY_COUNT=3

# 4. Function to download files with retry mechanism
download_with_retry() {
  local file_id="$1"
  local file_name="$2"
  local count=0

  until [ $count -ge $RETRY_COUNT ]
  do
    echo "Attempting to download $file_name (Attempt $((count + 1))/$RETRY_COUNT)..."
    conda run -n "$ENV_NAME" gdown --id "$file_id" -O "$file_name" && break
    count=$((count + 1))
    echo "Retry $count of $RETRY_COUNT for $file_name..."
    sleep 2  # Increased wait time to 2 seconds
  done

  if [ $count -ge $RETRY_COUNT ]; then
    error_exit "Failed to download $file_name after $RETRY_COUNT attempts."
  fi
}

# 5. Ensure Conda is installed
if ! command -v conda &> /dev/null; then
  error_exit "Conda is not installed. Please install Conda and try again."
fi

# 6. Initialize Conda for bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 7. Check if the environment exists
if conda info --envs | grep -w "^$ENV_NAME" > /dev/null 2>&1; then
  echo "Activating existing Conda environment: $ENV_NAME"
else
  echo "Creating new Conda environment: $ENV_NAME with Python 3.9"
  conda create -y -n "$ENV_NAME" python=3.9
fi

# 8. Activate the Conda environment
echo "Activating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# 9. Ensure gdown is installed in the environment
if ! conda run -n "$ENV_NAME" pip show gdown > /dev/null 2>&1; then
  echo "Installing gdown in the Conda environment..."
  conda run -n "$ENV_NAME" pip install gdown
fi

# 10. Determine the Operating System
OS=$(uname -s)

# 11. Install PyTorch, torchvision, torchaudio, and detectron2 based on OS and CUDA availability
install_dependencies() {
  if [[ "$OS" == "Darwin" ]]; then
    echo "Detected macOS. Installing PyTorch and torchvision for macOS..."
    conda run -n "$ENV_NAME" pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
    conda run -n "$ENV_NAME" pip install 'git+https://github.com/facebookresearch/detectron2.git'
  else
    # Check for NVIDIA GPU by looking for 'nvcc' or 'nvidia-smi'
    if command -v nvcc > /dev/null 2>&1 || command -v nvidia-smi > /dev/null 2>&1; then
      echo "CUDA detected. Installing GPU-supported PyTorch and torchvision..."
      conda run -n "$ENV_NAME" pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
      conda run -n "$ENV_NAME" pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
    else
      echo "No CUDA detected. Installing CPU-only PyTorch and torchvision..."
      conda run -n "$ENV_NAME" pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
      conda run -n "$ENV_NAME" pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
    fi
  fi
}

install_dependencies

# 12. Install additional Python dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
  echo "Installing additional Python dependencies from requirements.txt..."
  conda run -n "$ENV_NAME" pip install -r requirements.txt
else
  error_exit "requirements.txt not found in the current directory."
fi

# 13. Create models directory if it doesn't exist
FILEDIR=$(pwd)  # Set to current working directory
MODELS_DIR="$FILEDIR/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# 14. Download model files with retry mechanism
declare -A MODEL_FILES=(
  ["rcnn_bet365.pth"]="1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH"
  ["faster_rcnn.yaml"]="1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW"
  ["resnetv2_rgb_new.pth.tar"]="1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS"
  ["expand_targetlist.zip"]="1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I"
  ["domain_map.pkl"]="1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1"
)

for FILE_NAME in "${!MODEL_FILES[@]}"; do
  FILE_ID="${MODEL_FILES[$FILE_NAME]}"
  if [ -f "$FILE_NAME" ]; then
    echo "$FILE_NAME already exists. Skipping download."
  else
    download_with_retry "$FILE_ID" "$FILE_NAME"
  fi
done

# Unzip the file
unzip expand_targetlist.zip -d expand_targetlist

# Change to the extracted directory
cd expand_targetlist || exit 1  # Exit if the directory doesn't exist

# Check if there's a nested 'expand_targetlist/' directory
if [ -d "expand_targetlist" ]; then
  echo "Nested directory 'expand_targetlist/' detected. Moving contents up..."

  # Enable dotglob to include hidden files
  shopt -s dotglob

  # Move everything from the nested directory to the current directory
  mv expand_targetlist/* .

  # Disable dotglob to revert back to normal behavior
  shopt -u dotglob

  # Remove the now-empty nested directory
  rmdir expand_targetlist
  cd ../
else
  echo "No nested 'expand_targetlist/' directory found. No action needed."
fi

echo "Extraction completed successfully."



# 15. Final message
echo "All packages installed and models downloaded successfully!"
