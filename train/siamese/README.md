## Download the dataset
Download the Logo2k+ dataset
```commandline
pip install gdown
mkdir datasets
cd datasets
mkdir siamese_training
gdown --id 1IFDF7gyjnnyrns4Fm-Ui8sMloBsNY1EO -O Logo-2K+.zip
unzip Logo-2K+.zip -d Logo-2K+
gdown --id 1AkJP4E7Wki5miKd6DW8JlW8xf6UfU1ud -O List.zip
unzip List.zip -d List
```

Still in the datasets/siamese_training/ directory, download the Phishpedia reference list dataset
```commandline
gdown --id 1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I -O expand_targetlist.zip
unzip expand_targetlist.zip -d expand_targetlist 
gdown --id 1cuGAGe-HubaQWU8Gwn0evKSOake6hCTZ -O train_targets.txt
gdown --id 1GirhWiOVQpJWafhHA93elMfsUrxJzr9f -O test_targets.txt
```

## Prepare the label2id Dict
Run data.py to convert class names to class ids.

## Pretraining on Logo2k
This command runs the fine-tuning on the downloaded model:
```commandline
python -m train.siamese.train \
    --name logo2k \  # Name of this run. Used for monitoring and checkpointing.
    --model mobilenet_v2 \  # Which pretrained model to use.
    --logdir ./runs \  # Where to log training info.
    --dataset logo_2k \  # Name of custom dataset as specified and self-implemented above.
    --base_lr 0.01 # learning rate
```

## Finetuning on Reference List
Saving and utilizing the weights in the previous step, I finetune the model once again on our intended task:
```commandline
python -m train.siamese.train \
    --name targetlist_finetuned \  # Name of this run. Used for monitoring and checkpointing.
    --model mobilenet_v2 \  # Which pretrained model to use.
    --logdir ./runs \  # Where to log training info.
    --dataset targetlist \  # Name of custom dataset as specified and self-implemented above.
    --weights_path ./runs/logo2k/bit.pth.tar \ # Path to weights saved in the previous step, i.e. bit.pth.tar.
    --base_lr 0.01 # learning rate
```

## Export to onnx
Please refer to onnx.py for details.
