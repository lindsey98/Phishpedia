#!/bin/bash

# Source the Conda configuration
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create a new conda environment with Python 3.7
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
    conda create -n "$ENV_NAME" python=3.7
    conda activate "$ENV_NAME"
fi

mkl_path=$(conda info --base)/envs/"$ENV_NAME"/lib
echo "MKL path is $mkl_path"
# Export the LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$mkl_path"

# Get the CUDA and cuDNN versions, install pytorch, torchvision
pip install -r requirements.txt
conda install typing_extensions

# Check if nvcc (CUDA compiler) is available
if ! command -v nvcc &> /dev/null; then
    echo "CUDA is not available."
    pip install torch==1.8.1 torchvision
    # Install Detectron2
    python -m --user pip install 'git+https://github.com/facebookresearch/detectron2.git'
else
    cuda_version=$(nvcc --version | grep release | awk '{print $6}' | cut -c2- | awk -F. '{print $1"."$2}')
    pip install torch==1.8.1 torchvision -f "https://download.pytorch.org/whl/cu${cuda_version//.}/torch_stable.html"
    # Install Detectron2
    cuda_version=$(nvcc --version | grep release | awk '{print $6}' | cut -c2- | awk -F. '{print $1$2}')
    case $cuda_version in
        "111" | "102" | "101")
          python -m pip install detectron2 -f \
      https://dl.fbaipublicfiles.com/detectron2/wheels/cu"$cuda_version"/torch1.8/index.html
        ;;
        *)
          echo "Please build Detectron2 from source https://detectron2.readthedocs.io/en/latest/tutorials/install.html">&2
          exit 1
          ;;
    esac
fi


## Download models
export LD_LIBRARY_PATH=""
pip install git+https://github.com/lindsey98/Phishpedia.git
package_location=$(pip show phishpedia | grep Location | awk '{print $2}')

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
  
  cd ../../siamese_pedia/
  file_id="1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I"
  output_file="expand_targetlist.zip"
  # Download the file using wget
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O "$output_file" && rm -rf /tmp/cookies.txt
  dir_name=$(unzip -l expand_targetlist.zip | awk '/^[^ ]/ {print $4}' | awk -F'/' '{print $1}' | uniq)
  echo $dir_name
  
  file_id="1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1"
  output_file="domain_map.pkl"
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O "$output_file" && rm -rf /tmp/cookies.txt
    
  file_id="1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS"
  output_file="resnetv2_rgb_new.pth.tar"
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O "$output_file" && rm -rf /tmp/cookies.txt
  
fi

echo "All packages installed successfully!"
