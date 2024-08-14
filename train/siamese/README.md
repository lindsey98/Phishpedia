## Prepare the label2id Dict
Run data.py to convert class names to class ids.

## Pretraining on Logo2k
This command runs the fine-tuning on the downloaded model:
```
python -m train.siamese.train \
    --name logo2k \  # Name of this run. Used for monitoring and checkpointing.
    --model mobilenet_v2 \  # Which pretrained model to use.
    --logdir ./runs \  # Where to log training info.
    --dataset logo_2k \  # Name of custom dataset as specified and self-implemented above.
    --base_lr 0.01 # learning rate
```

## Finetuning on Reference List
Saving and utilizing the weights in the previous step, I finetune the model once again on our intended task:
```
python -m train.siamese.train \
    --name targetlist_finetuned \  # Name of this run. Used for monitoring and checkpointing.
    --model mobilenet_v2 \  # Which pretrained model to use.
    --logdir ./runs \  # Where to log training info.
    --dataset targetlist \  # Name of custom dataset as specified and self-implemented above.
    --weights_path ./runs/logo2k/bit.pth.tar \ # Path to weights saved in the previous step, i.e. bit.pth.tar.
    --base_lr 0.01 # learning rate
```