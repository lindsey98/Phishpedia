#!/bin/bash

FILEDIR=$(pwd)
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda info --envs | grep -w "myenv" > /dev/null

if [ $? -eq 0 ]; then
   echo "Activating Conda environment myenv"
   conda activate myenv
else
   echo "Creating and activating new Conda environment $ENV_NAME with Python 3.8"
   conda create -n myenv python=3.8
   conda activate myenv
fi

pip install -r requirements.txt

OS=$(uname -s)

if [[ "$OS" == "Darwin" ]]; then
  echo "Installing PyTorch and torchvision for macOS."
  pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
  python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
else
  # Check if NVIDIA GPU is available for Linux and Windows
  if command -v nvcc &> /dev/null; then
    echo "CUDA is detected, installing GPU-supported PyTorch and torchvision."
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
  else
    echo "No CUDA detected, installing CPU-only PyTorch and torchvision."
    pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
  fi
fi

## Download models
pip install -v .
package_location=$(pip show phishpedia | grep Location | awk '{print $2}')

if [ -z "$package_location" ]; then
  echo "Package Phishpedia not found in the Conda environment myenv."
  exit 1
else
  echo "Going to the directory of package Phishpedia in Conda environment myenv."
  cd "$package_location/phishpedia/src/detectron2_pedia/output/rcnn_2" || exit
  pip install gdown
  gdown --id 1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH
  cd "$package_location/phishpedia/src/siamese_pedia/" || exit
  gdown --id 1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS
  gdown --id 1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I
  gdown --id 1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1
fi

# Replace the placeholder in the YAML template

sed "s|CONDA_ENV_PATH_PLACEHOLDER|$package_location/phishpedia|g" "$FILEDIR/phishpedia/configs_template.yaml" > "$FILEDIR/phishpedia/configs.yaml"

echo "All packages installed successfully!"
