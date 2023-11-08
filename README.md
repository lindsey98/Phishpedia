# Phishpedia A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages

<div align="center">

![Dialogues](https://img.shields.io/badge/Proctected\_Brands\_Size-277-green?style=flat-square)
![Dialogues](https://img.shields.io/badge/Phishing\_Benchmark\_Size-30k-green?style=flat-square)


</div>
<p align="center">
  <a href="https://www.usenix.org/conference/usenixsecurity21/presentation/lin">Paper</a> •
  <a href="https://sites.google.com/view/phishpedia-site/">Website</a> •
  <a href="https://www.youtube.com/watch?v=ZQOH1RW5DmY">Video</a> •
   <a href="https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view?usp=drive_link">Dataset</a> •
  <a href="#citation">Citation</a>
</p>

- This is the official implementation of "Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages" USENIX'21 [link to paper](https://www.usenix.org/conference/usenixsecurity21/presentation/lin), [link to our website](https://sites.google.com/view/phishpedia-site/), [link to our dataset](https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view?usp=drive_link).

- Existing reference-based phishing detectors:
  - :x: Lack of interpretability
  - :x: Lack of generalization performance in the wild
  - :x: Lack of a large-scale phishing benchmark dataset
    
- The contributions of our paper:
   - :white_check_mark: We propose a phishing identification system Phishpedia, which has high identification accuracy and low runtime overhead, outperforming the relevant state-of-the-art identification approaches. 
   - :white_check_mark: Our system provides explainable annotations which increase users' confidence in model prediction
   - :white_check_mark: We conducted a phishing discovery experiment on emerging domains fed from CertStream and discovered 1,704 real phishing, out of which 1133 are zero-days   

## Framework
    
<img src="phishpedia/big_pic/overview.png" style="width:2000px;height:350px"/>

```Input```: A URL and its screenshot ```Output```: Phish/Benign, Phishing target
- Step 1: Enter <b>Deep Object Detection Model</b>, get predicted logos and inputs (inputs are not used for later prediction, just for explanation)

- Step 2: Enter <b>Deep Siamese Model</b>
    - If Siamese report no target, ```Return  Benign, None```
    - Else Siamese report a target, ```Return Phish, Phishing target``` 
    
## Project structure
```
- src
    - adv_attack: adversarial attacking scripts
    - detectron2_pedia: training script for object detector
     |_ output
      |_ rcnn_2
        |_ rcnn_bet365.pth 
    - siamese_pedia: inference script for siamese
     |_ siamese_retrain: training script for siamese
     |_ expand_targetlist
         |_ 1&1 Ionos
         |_ ...
     |_ domain_map.pkl
     |_ resnetv2_rgb_new.pth.tar
    - siamese.py: main script for siamese
    - pipeline_eval.py: evaluation script for general experiment

- tele: telegram scripts to vote for phishing 
- phishpedia_config.py: config script for phish-discovery experiment 
- phishpedia_main.py: main script for phish-discovery experiment 
```

## Instructions
Requirements: 
- CUDA 11

1. Create a local clone of Phishpedia
```
git clone https://github.com/lindsey98/Phishpedia.git
```

2. Setup
```
cd Phishpedia
chmod +x ./setup.sh
./setup.sh
```
If you encounter any problem in downloading the models, you can manually download them from here https://huggingface.co/Kelsey98/Phishpedia. And put them into the corresponding conda environment.

3. 
```
conda activate myenv
```

Run in Python to test a single website

```python
from phishpedia.phishpedia_main import test
import matplotlib.pyplot as plt
from phishpedia.phishpedia_config import load_config

url = open("phishpedia/datasets/test_sites/accounts.g.cdcde.com/info.txt").read().strip()
screenshot_path = "phishpedia/datasets/test_sites/accounts.g.cdcde.com/shot.png"
cfg_path = None # None means use default config.yaml
ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(cfg_path)

phish_category, pred_target, plotvis, siamese_conf, pred_boxes = test(url, screenshot_path,
                                                                      ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)

print('Phishing (1) or Benign (0) ?', phish_category)
print('What is its targeted brand if it is a phishing ?', pred_target)
print('What is the siamese matching confidence ?', siamese_conf)
print('Where is the predicted logo (in [x_min, y_min, x_max, y_max])?', pred_boxes)
plt.imshow(plotvis[:, :, ::-1])
plt.title("Predicted screenshot with annotations")
plt.show()
```
Or run in the terminal to test a list of sites, copy run.py to your local machine and run
```
python run.py --folder <folder you want to test e.g. phishpedia/datasets/test_sites> --results <where you want to save the results e.g. test.txt> --no_repeat
```


## Miscellaneous
- In our paper, we also implement several phishing detection and identification baselines, see [here](https://github.com/lindsey98/PhishingBaseline)
- The logo targetlist described in our paper includes 181 brands, we have further expanded the targetlist to include 277 brands in this code repository 
- For the phish discovery experiment, we obtain feed from [Certstream phish_catcher](https://github.com/x0rz/phishing_catcher), we lower the score threshold to be 40 to process more suspicious websites, readers can refer to their repo for details
- We use Scrapy for website crawling [Repo here](https://github.com/lindsey98/MyScrapy.git) 

## Citation 
If you find our work useful in your research, please consider citing our paper by:

```bibtex
@inproceedings{lin2021phishpedia,
  title={Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages},
  author={Lin, Yun and Liu, Ruofan and Divakaran, Dinil Mon and Ng, Jun Yang and Chan, Qing Zhou and Lu, Yiwen and Si, Yuxuan and Zhang, Fan and Dong, Jin Song},
  booktitle={30th $\{$USENIX$\}$ Security Symposium ($\{$USENIX$\}$ Security 21)},
  year={2021}
}
```

## Contacts
If you have any issues running our code, you can raise an issue or send an email to liu.ruofan16@u.nus.edu, lin_yun@sjtu.edu.cn, and dcsdjs@nus.edu.sg
