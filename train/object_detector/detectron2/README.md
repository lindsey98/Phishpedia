#### To run evaluation for a trained model
```commandline
python train/object_detector/detectron2/inference.py \
    --eval-only \
    --config-file train/object_detector/detectron2/configs/faster_rcnn.yaml \
    MODEL.WEIGHTS models/rcnn.pth  # Path to trained checkpoint
```