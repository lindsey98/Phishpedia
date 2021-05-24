# Phishpedia A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages

- This is the official implementation of "Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages" USENIX'21 [[paper](https://www.usenix.org/conference/usenixsecurity21/presentation/lin)]
    
## Framework
    
<img src="big_pic/pic.png" style="width:2000px;height:350px"/>

```Input```: A URL and its screenshot ```Output```: Phish/Benign, Phishing target
- Step 1: Enter <b>Deep Object Detection Model</b>, get predicted elements

- Step 2: Enter <b>Deep Siamese Model</b>
    - If Siamese report no target, ```Return  Benign, None```
    - Else Siamese report a target, ```Return Phish, Phishing target``` 
    
## Project structure
```
- src
    - siamese_retrain: training script for siamese
    - detectron2_peida: training script for object detector
    - phishpedia: inference script for siamese
    - util: other scripts (chromedriver utilities)
    - siamese.py: main script for siamese
    - pipeline_eval.py: evaluation script for general experiment

- phishpedia_config.py: config script for phish-discovery experiment 
- phishpedia_main.py: main script for phish-discovery experiment 
```


       
## Requirements
- Linux machine equipped with GPU 
python=3.7
torch=1.5.1
torchvision=0.6.0
- Run
```
pip install -r requirements.txt
```
- Install Detectron2 manually, see the official installation [[guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)]. 

## Instructions
1. Download all the model files:
- First download [[Siamese model weights](https://drive.google.com/file/d/1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS/view?usp=sharing)],
[[Logo targetlist](https://drive.google.com/file/d/1_C8NSQYWkpW_-tW8WzFaBr8vDeBAWQ87/view?usp=sharing)],
[[Brand domain dictionary](https://drive.google.com/file/d/1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1/view?usp=sharing)], put them under **src/phishpedia**

- Then download [[Object detector weights](https://drive.google.com/file/d/1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH/view?usp=sharing)],
put it under **src/detectron2_pedia/output/rcnn_2**

2. Download all data files
- Download [[phish 30k](https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view?usp=sharing)], 
[[benign30k](https://drive.google.com/file/d/1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_/view?usp=sharing)] dataset,
unzip and move them to **datasets/**

3. Run experiment 
- For phish discovery experiment, the data folder should be organized in [[this format](https://github.com/lindsey98/Phishpedia/tree/main/datasets/test_sites)]:
```
python phishpedia_main.py --folder [data folder you want to test] --results [xxx.txt]
```
- For general experiment on phish30k and benign30k: 
please run evaluation scripts
```
python -m src.pipeline_eval --data-dir datasets/phish_sample_30k --mode phish --write-txt output_phish.txt --ts [threshold for siamese, 0.83 is suggested]
python -m src.pipeline_eval --data-dir datasets/benign_sample_30k --mode benign --write-txt output_benign.txt --ts [threshold for siamese, 0.83 is suggested]
```

## Training the model (Optional)
1. If you want to train object detection faster-rcnn model yourself, 
- First dowonload training [[data](https://drive.google.com/file/d/1L3KSWEXcnWzYdJ4hPrNEUvC8jaaNOiBa/view?usp=sharing)] to **datasets/**

- Second step is to create folder to save trained weights and log:
```
mkdir src/detectron2_pedia/output
```
- Then start training 
To train on a single gpu:
```
python -m src.detectron2_pedia.train_net \
       --config-file src/detectron2_pedia/configs/faster_rcnn.yaml
```

To train on multiple gpus:
```
python -m src.detectron2_pedia.train_net \
       --num-gpus 4 \
       --config-file src/detectron2_pedia/configs/faster_rcnn.yaml
```

To resume training from a checkpoint (finds last checkpoint from cfg.OUTPUT_DIR)
```
python -m src.detectron2_pedia.train_net \
       --num-gpus 4 \
       --config-file src/detectron2_pedia/configs/faster_rcnn.yaml \
       --resume
```
- Launch [[DAG](http://openaccess.thecvf.com/content_ICCV_2017/papers/Xie_Adversarial_Examples_for_ICCV_2017_paper.pdf)] adversarial attack on Faster-RCNN:
```
python -m src.detectron2_pedia.run_DAG \
    --cfg-path src/detectron2_pedia/configs/faster_rcnn.yaml \
    --weights-path src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth \
    --results-save-path coco_instances_results.json \
    --vis-save-dir saved
```

2. If you want to train siamese
- I first pretrained on the Logos2k [[download here](https://drive.google.com/file/d/1gniiDM0mgwIzE4t1svWXLI5-A5AJgVlh/view?usp=sharing)] dataset, using a pretrained BiT-M ResNet50x1 model, which we have to download first:
```
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz # download pretraind weights
```
- This command runs the pre-training on the downloaded model:
```
python -m src.siamese_retrain.bit_pytorch.train \
    --name {exp_name} \  # Name of this run. Used for monitoring and checkpointing.
    --model BiT-M-R50x1 \  # Which pretrained model to use.
    --logdir {log_dir} \  # Where to log training info.
    --dataset logo_2k \  # Name of custom dataset as specified and self-implemented above.
```
- Saving and utilizing the weights in the previous step, I finetune the model on our logo targetlist dataset:
Download [[siamese training list](https://drive.google.com/file/d/1cuGAGe-HubaQWU8Gwn0evKSOake6hCTZ/view?usp=sharing)], 
[[siamese testing list](https://drive.google.com/file/d/1GirhWiOVQpJWafhHA93elMfsUrxJzr9f/view?usp=sharing)],
[[siamese datadict](https://drive.google.com/file/d/12GjdcYeSBbPji8pCq5KrFhWmqUC451Pc/view?usp=sharing)],
put them under **src/siamese_retrain**.
Run
```
python -m src.siamese_retrain.bit_pytorch.train \
    --name {exp_name} \  # Name of this run. Used for monitoring and checkpointing.
    --model BiT-M-R50x1 \  # Which pretrained model to use.
    --logdir {log_dir} \  # Where to log training info.
    --dataset targetlist \  # Name of custom dataset as specified and self-implemented above.
    --weights_path {weights_path} \  # Path to weights saved in the previous step, i.e. bit.pth.tar.
```
- Launch adversarial attack ([[i-FGSM](https://arxiv.org/pdf/1412.6572.pdf))], [[i-StepLL](https://arxiv.org/pdf/1611.01236.pdf)], [[DeepFool](https://arxiv.org/pdf/1511.04599.pdf)], [[C&W L2](https://arxiv.org/pdf/1608.04644.pdf)], [[BPDA with Linf-PGD](https://arxiv.org/pdf/1802.00420.pdf)]) on siamese:
Run src/adv_attack/gradient masking siamese.ipynb 

 