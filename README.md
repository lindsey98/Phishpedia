# Memory efficient Phishpedia


## Setup instructions
1. Install torch, torchvision that are compatible with your CUDA. For your reference, I am using torch==2.4.0, torchvision==0.19.0.

2. Install the requirements.txt
```
conda activate [your_env_name]
pip install -r requirements.txt
```

## Download the pretrained models
```commandline
pip install gdown
mkdir models/
cd models
gdown --id 1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1 -O domain_map.pkl
gdown --id 17anjM7tdDOVkOnpCy_I-dmAmBknYeJHJ -O yolo_nano_320.onnx
gdown --id 1HKefBeT7VDiVxgPx06Qyki5c93_xN89X -O mobilenetv2_64.onnx
cd ../
gdown --id 1N76ehGTI45TC2paUNRygn0VZzNjT8QG8 -O LOGO_FEATS.npy
gdown --id 1B-W7h9h1n_na7Q6e87sPfdq1F9C-Y5nY -O LOGO_FILES.npy
```

## The expected directory structure would be:
```
models/
|– domain_map.pkl
|– yolo_nano_320.onnx
|– mobilenetv2_64.onnx
configs.py # model loading logics
configs.yaml # specify the model paths, hyperparameter configurations
logo_matching.py # siamese matching logics
phishpedia.py # main phishpedia code
LOGO_FEATS.npy # cached logo embeddings
LOGO_FILES.npy # ground-truth brands for those logo embeddings
utils.py # other utils 
```

## Run Phishpedia, the inference are done on cpu
```commandline
python phishpedia.py --folder [folder_to_test]
```

### Training scripts
The training scripts are in train/object_detector and train/siamese