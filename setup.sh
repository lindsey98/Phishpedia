#!/bin/bash

# # Set Conda environment as an environment variable
# # export MYENV=$(conda info --base)/envs/"$ENV_NAME"

# # Get the CUDA and cuDNN versions, install pytorch, torchvision
# conda run -n "$ENV_NAME" pip install -r requirements.txt

# # Install pytorch, torchvision, detectron2
# if command -v nvcc &> /dev/null; then
#    conda run -n "$ENV_NAME" pip install torch==1.9.0 torchvision -f "https://download.pytorch.org/whl/cu111/torch_stable.html"
#    conda run -n "$ENV_NAME" python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
# else
#    conda run -n "$ENV_NAME" pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
#    conda run -n "$ENV_NAME" python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
# fi

# ## Download models
# conda run -n "$ENV_NAME" pip install -v .
# package_location=$(conda run -n "$ENV_NAME" pip show phishpedia | grep Location | awk '{print $2}')

FILEDIR=$(pwd)
conda info --envs | grep -w "$ENV_NAME" > /dev/null

if [ $? -eq 0 ]; then
   echo "Activating Conda environment myenv"
   conda activate myenv
else
   echo "Creating and activating new Conda environment $ENV_NAME with Python 3.8"
   conda create -n myenv python=3.8
   conda activate myenv
fi

pip install -r requirements.txt

# Install pytorch, torchvision, detectron2
if command -v nvcc &> /dev/null; then
   pip install torch==1.9.0 torchvision -f "https://download.pytorch.org/whl/cu111/torch_stable.html"
   python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
else
   pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
   python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
fi

## Download models
pip install -v .
package_location=$(pip show phishpedia | grep Location | awk '{print $2}')

if [ -z "Phishpedia" ]; then
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
