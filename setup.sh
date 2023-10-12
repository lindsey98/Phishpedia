#!/bin/bash

FILEDIR=$(pwd)

# Source the Conda configuration
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="myenv"

# Check if the environment already exists
conda info --envs | grep -w "$ENV_NAME" > /dev/null

if [ $? -eq 0 ]; then
   echo "Activating Conda environment $ENV_NAME"
   conda activate "$ENV_NAME"
else
   echo "Creating and activating new Conda environment $ENV_NAME with Python 3.8"
   conda create -n "$ENV_NAME" python=3.8
   conda activate "$ENV_NAME"
fi

# Set Conda environment as an environment variable
export MYENV=$(conda info --base)/envs/"$ENV_NAME"

# Get the CUDA and cuDNN versions, install pytorch, torchvision
conda run -n "$ENV_NAME" pip install -r requirements.txt

conda run -n "$ENV_NAME" pip install torch==1.9.0 torchvision -f \
  "https://download.pytorch.org/whl/cu111/torch_stable.html"

conda run -n "$ENV_NAME" python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html


## Download models
conda run -n "$ENV_NAME" pip install -v .
package_location=$(conda run -n "$ENV_NAME" pip show phishpedia | grep Location | awk '{print $2}')

if [ -z "Phishpedia" ]; then
  echo "Package Phishpedia not found in the Conda environment myenv."
  exit 1
else
  echo "Going to the directory of package Phishpedia in Conda environment myenv."
  cd "$package_location/phishpedia/src/detectron2_pedia/" || exit
  pip install gdown
  gdown --id 1eKVEGnAznFktm5s0plKjnwpUMGZfK9gX
  cd "$package_location/phishpedia/src/siamese_pedia/" || exit
  gdown --id 11LZBxv4SIKQbqh2hcuZaQ-m00Tl1Mhkc
fi

# Replace the placeholder in the YAML template

sed "s|CONDA_ENV_PATH_PLACEHOLDER|$package_location/phishpedia|g" "$FILEDIR/phishpedia/configs_template.yaml" > "$FILEDIR/phishpedia/configs.yaml"


echo "All packages installed successfully!"
