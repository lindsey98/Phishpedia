#!/bin/bash

# Check if ENV_NAME is set
if [ -z "$ENV_NAME" ]; then
  echo "ENV_NAME is not set. Please set the environment name and try again."
  exit 1
fi

retry_count=3  # Number of retries

download_with_retry() {
  local file_id=$1
  local file_name=$2
  local count=0

  until [ $count -ge $retry_count ]
  do
    conda run -n "$ENV_NAME" gdown --id "$file_id" -O "$file_name" && break  # attempt to download and break if successful
    count=$((count+1))
    echo "Retry $count of $retry_count..."
    sleep 1  # wait for 1 second before retrying
  done

  if [ $count -ge $retry_count ]; then
    echo "Failed to download $file_name after $retry_count attempts."
    exit 1
  fi
}

FILEDIR=$(pwd)
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda info --envs | grep -w "phishpedia" > /dev/null

if [ $? -eq 0 ]; then
   echo "Activating Conda environment phishpedia"
   conda activate "$ENV_NAME"
else
   echo "Creating and activating new Conda environment phishpedia with Python 3.8"
   conda create -n "$ENV_NAME" python=3.8
   conda activate "$ENV_NAME"
fi


OS=$(uname -s)

if [[ "$OS" == "Darwin" ]]; then
  echo "Installing PyTorch and torchvision for macOS."
  conda run -n "$ENV_NAME" pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
  conda run -n "$ENV_NAME"  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
else
  # Check if NVIDIA GPU is available for Linux and Windows
  if command -v nvcc || command -v nvidia-smi &> /dev/null; then
    echo "CUDA is detected, installing GPU-supported PyTorch and torchvision."
    conda run -n "$ENV_NAME" pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    conda run -n "$ENV_NAME" python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
  else
    echo "No CUDA detected, installing CPU-only PyTorch and torchvision."
    conda run -n "$ENV_NAME" pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    conda run -n "$ENV_NAME" python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
  fi
fi

conda run -n "$ENV_NAME" pip install -r requirements.txt

## Download models
echo "Going to the directory of package Phishpedia in Conda environment myenv."
mkdir -p models/
cd models/


# RCNN model weights
if [ -f "rcnn_bet365.pth" ]; then
  echo "RCNN model weights exists... skip"
else
  download_with_retry 1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH rcnn_bet365.pth
fi

# Faster RCNN config
if [ -f "faster_rcnn.yaml" ]; then
  echo "RCNN model config exists... skip"
else
  download_with_retry 1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW faster_rcnn.yaml
fi

# Siamese model weights
if [ -f "resnetv2_rgb_new.pth.tar" ]; then
  echo "Siamese model weights exists... skip"
else
  download_with_retry 1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS resnetv2_rgb_new.pth.tar
fi

# Reference list
if [ -f "expand_targetlist.zip" ]; then
  echo "Reference list exists... skip"
else
  download_with_retry 1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I expand_targetlist.zip
fi

# Domain map
if [ -f "domain_map.pkl" ]; then
  echo "Domain map exists... skip"
else
  download_with_retry 1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1 domain_map.pkl
fi



# Replace the placeholder in the YAML template
echo "All packages installed successfully!"
