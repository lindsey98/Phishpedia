
## Download the training dataset from Google drive
```commandline
pip install gdown
mkdir datasets
cd datasets
gdown --id 1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_ -O benign_sample_30k.zip
mkdir object_detector_training
cd object_detector_training
gdown --id 1u56I0IHBgM9glNJl2wcLfaihp1L_U7eD -O coco_train.json
gdown --id 1bH3Yp6K1B37B_sS_MNMz7yvYcOhOu-J8 -O coco_test.json
```

## Prepare the dataset
Run prepare_data.py to prepare the data into expected format.
The expected dataset need to be organized into the following format
```
custom_dataset
|– images/
|   |– train/
|   |   |– image1_t.jpg
|   |   |– image2_t.jpg
|   |– val/
|   |   |– image1_v.jpg
|   |   |– image2_v.jpg
    
|– labels
|   |– train/
|   |   |– image1_t.txt
|   |   |– image2_t.txt
|   |– val/
|   |   |– image1_v.txt
|   |   |– image2_v.txt
```

And each label.txt is in the following format
```
{labelid} {x_center} {y_center} {w} {h}
{labelid} {x_center} {y_center} {w} {h} ...
```

Create a config.yaml, put it in the same parent directory as the images/, in the config.yaml, put
```yaml
train: images/train
val: images/val
nc: 2 # number of classes, e.g. 2
names: [
 'class name 1',
 'class name 2',
 ......
]
```


## Training the yolov8
In train.py, yolov8n.pt stands for yolov8-nano, and yolov8s.pt stands for yolov8-small.
For example, to train yolov8-nano with image size of 320 for 20 epochs:
```commandline
python -m train.object_detector.train 
--data ./datasets/object_detector_training/config.yaml 
--base_model yolov8n.pt 
--name yolo_nano_320
-epochs 20
--imgsz 320
--device 0
```

## Convert trained yolov8 to onnx
Please refer to the onnx.py.

