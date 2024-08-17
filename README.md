# Memory efficient Phishpedia


## Setup instructions
1. Install torch, torchvision that are compatible with your CUDA. For your reference, I am using torch==2.4.0, torchvision==0.19.0.

2. Install the requirements.txt
```
conda activate [your_env_name]
pip install -r requirements.txt
```

## Download the pretrained models from Google drive
```commandline
pip install gdown
mkdir models/
cd models
gdown --id 1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1 -O domain_map.pkl
gdown --id 1eDpSOieSBrchhcvA_U4UoEZRWioJ3VhQ -O yolo_nano_640.onnx 
gdown --id 1zg5Gw2SMB6xOMtX4yQHsD8j__ZZDQze6 -O mobilenetv2_128.onnx
gdown --id 1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I -O expand_targetlist.zip
unzip expand_targetlist.zip -d expand_targetlist
```

## The expected directory structure would be:
```
models/
|– domain_map.pkl
|– yolo_nano_640.onnx
|– mobilenetv2_128.onnx
|- expand_targetlist/
   |- Adobe/
   |- Amazon/
   ...
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

### Re-training scripts
All the training scripts are in [train/object_detector](train/object_detector/README.md) and [train/siamese](train/siamese/README.md).

### Benchmarking Phishpedia on Phish30k and Benign30k datasets
1. Download the 30k benign + 30k phishing datasets
```commandline
mkdir datasets
cd datasets/
gdown --id 1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_ -O benign_sample_30k.zip
unzip benign_sample_30k.zip -d benign_sample_30k
gdown --id 12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g -O phish_sample_30k.zip
unzip phish_sample_30k.zip -d phish_sample_30k
```

2. Run the benchmarking
```commandline
python -m train.benchmark --mode benign
python -m train.benchmark --mode phish
```

3. Compute the metrics False Positive Rate and Recall