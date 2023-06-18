#!/bin/bash

FILEDIR=$(pwd)

# Source the Conda configuration
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# # Create a new conda environment with Python 3.7
ENV_NAME="myenv"

# Check if the environment already exists
conda info --envs | grep -w "$ENV_NAME" > /dev/null

if [ $? -eq 0 ]; then
   # If the environment exists, activate it
   echo "Activating Conda environment $ENV_NAME"
   conda activate "$ENV_NAME"
else
   # If the environment doesn't exist, create it with Python 3.7 and activate it
   echo "Creating and activating new Conda environment $ENV_NAME with Python 3.7"
   conda create -n "$ENV_NAME" python=3.8
   conda activate "$ENV_NAME"
fi

# Set Conda environment as an environment variable
export MYENV=$(conda info --base)/envs/"$ENV_NAME"

# Get the CUDA and cuDNN versions, install pytorch, torchvision
conda run -n "$ENV_NAME" pip install -r requirements.txt
conda run -n "$ENV_NAME" pip install cryptography==38.0.4
conda install typing_extensions
conda run -n "$ENV_NAME" pip install torch==1.9.0 torchvision -f \
  "https://download.pytorch.org/whl/cu111/torch_stable.html"

conda run -n "$ENV_NAME" python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html


## Download models
export LD_LIBRARY_PATH=""
conda run -n "$ENV_NAME" pip install git+https://github.com/lindsey98/Phishpedia.git
package_location=$(conda run -n myenv pip show phishpedia | grep Location | awk '{print $2}')

if [ -z "Phishpedia" ]; then
  echo "Package Phishpedia not found in the Conda environment myenv."
  exit 1
else
  echo "Going to the directory of package Phishpedia in Conda environment myenv."
  cd "$package_location/phishpedia" || exit
  cd src/detectron2_pedia/output/rcnn_2/
  file_id="1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH"
  output_file="rcnn_bet365.pth"
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O "$output_file" && rm -rf /tmp/cookies.txt
  
  cd "$package_location/phishpedia" || exit
  cd src/siamese_pedia/
  file_id="1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I"
  output_file="expand_targetlist.zip"
  # Download the file using wget
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O "$output_file" && rm -rf /tmp/cookies.txt
  dir_name=$(unzip -l expand_targetlist.zip | awk '/^[^ ]/ {print $4}' | awk -F'/' '{print $1}' | uniq)
  echo $dir_name
  
  file_id="1nTIC6311dvdY4cGsrI4c3WMndSauuHSm"
  output_file="domain_map.pkl"
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O "$output_file" && rm -rf /tmp/cookies.txt
    
  file_id="1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS"
  output_file="resnetv2_rgb_new.pth.tar"
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O "$output_file" && rm -rf /tmp/cookies.txt
  
fi

# Replace the placeholder in the YAML template

sed "s|CONDA_ENV_PATH_PLACEHOLDER|$package_location/phishpedia|g" "$FILEDIR/phishpedia/configs_template.yaml" > "$FILEDIR/phishpedia/configs.yaml"


echo "All packages installed successfully!"
