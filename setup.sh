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

## Download models
export LD_LIBRARY_PATH=""
pip install git+https://github.com/lindsey98/Phishpedia.git
package_location=$(pip show phishpedia | grep Location | awk '{print $2}')

if [ -z "Phishpedia" ]; then
  echo "Package Phishpedia not found in the Conda environment myenv."
  exit 1
else
  echo "Going to the directory of package PhishIntention in Conda environment myenv."
  cd "$package_location/phishpedia" || exit
  pwd
  
fi

echo "All packages installed successfully!"
