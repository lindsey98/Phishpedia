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
  - :x: Lack of **interpretability**, only give binary decision (legit or phish)
  - :x: **Not robust against distribution shift**, because the classifier is biased towards the phishing training set
  - :x: **Lack of a large-scale phishing benchmark** dataset
    
- The contributions of our paper:
   - :white_check_mark: We propose a phishing identification system Phishpedia, which has high identification accuracy and low runtime overhead, outperforming the relevant state-of-the-art identification approaches. 
   - :white_check_mark: We are the first to propose to use **consistency-based method** for phishing detection, in place of the traditional classification-based method. We investigate the consistency between the webpage domain and its brand intention. The detected brand intention provides a **visual explanation** for phishing decision.
   - :white_check_mark: Phishpedia is **NOT trained on any phishing dataset**, addressing the potential test-time distribution shift problem.
   - :white_check_mark: We release a **30k phishing benchmark dataset**, each website is annotated with its URL, HTML, screenshot, and target brand: https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view?usp=drive_link.
   - :white_check_mark: We set up a **phishing monitoring system**, investigating emerging domains fed from CertStream, and we have discovered 1,704 real phishing, out of which 1133 are zero-days not reported by industrial antivirus engine (Virustotal).  

## Framework
    
<img src="./datasets/overview.png" style="width:2000px;height:350px"/>

```Input```: A URL and its screenshot ```Output```: Phish/Benign, Phishing target
- Step 1: Enter <b>Deep Object Detection Model</b>, get predicted logos and inputs (inputs are not used for later prediction, just for explanation)

- Step 2: Enter <b>Deep Siamese Model</b>
    - If Siamese report no target, ```Return  Benign, None```
    - Else Siamese report a target, ```Return Phish, Phishing target``` 

## Project structure
:pushpin: We need to move everything under expand_targetlist/expand_targetlist to expand_targetlist/ so that there are no nested directories.
```
- models/
|___ rcnn_bet365.pth
|___ faster_rcnn.yaml
|___ resnetv2_rgb_new.pth.tar
|___ expand_targetlist/
  |___ Adobe/
  |___ Amazon/
  |___ ......
|___ domain_map.pkl
- logo_recog.py: Deep Object Detection Model
- logo_matching.py: Deep Siamese Model 
- configs.yaml: Configuration file
- phishpedia.py: Main script
```

## Instructions

Prerequisite: [Anaconda installed](https://docs.anaconda.com/free/anaconda/install/index.html) 

<details>
  <summary>Running Inference from the Command Line</summary>

- Step 1. Create a local clone of Phishpedia, and setup the phishpedia conda environment.
In this step, we would be installing the core dependencies of Phishpedia such as pytorch, and detectron2.
In addition, we would also download the model checkpoints and brand reference list.
This step may take some time.
```bash
git clone https://github.com/lindsey98/Phishpedia.git
cd Phishpedia
chmod +x ./setup.sh
./setup.sh
```

- Step 2. Activate conda environment _phishpedia_:
```bash
conda activate phishpedia
```

- Step 3. Run in bash 
```bash
python phishpedia.py --folder <folder you want to test e.g. ./datasets/test_sites>
```

The testing folder should be in the structure of:

```
test_site_1
|__ info.txt (Write the URL)
|__ shot.png (Save the screenshot)
test_site_2
|__ info.txt (Write the URL)
|__ shot.png (Save the screenshot)
......
```
</details>

<details>
  <summary>Running Phishpedia as a GUI tool (PyQt5-based)</summary>
  
  Refer to [GUItool/](GUItool/)
</details>

<details>
  <summary>Running Phishpedia as a GUI tool (web-browser-based)</summary>
  
  Refer to [WEBtool/](WEBtool/)
</details>

<details>
  <summary>Running Phishpedia as a Chrome plugin</summary>
  
  Refer to [Plugin_for_Chrome/](Plugin_for_Chrome/)
</details>
  
## Miscellaneous
- In our paper, we also implement several phishing detection and identification baselines, see [here](https://github.com/lindsey98/PhishingBaseline)
- The logo targetlist described in our paper includes 181 brands, we have further expanded the targetlist to include 277 brands in this code repository 
- For the phish discovery experiment, we obtain feed from [Certstream phish_catcher](https://github.com/x0rz/phishing_catcher), we lower the score threshold to be 40 to process more suspicious websites, readers can refer to their repo for details
- We use Scrapy for website crawling 

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
