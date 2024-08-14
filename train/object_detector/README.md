
## Prepare the dataset
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
{labelid} {x_center} {y_center} {w} {h}
```

Create a config.yaml, put it in the same parent directory as the images/, in the config.yaml, put
```yaml
train: images/train
val: images/val
nc: # number of classes, e.g. 2
names: [
 'class name 1',
 'class name 2',
 ......
]
```

## Training the yolov8
In train.py, yolov8n.pt stands for yolov8-nano, and yolov8s.pt stands for yolov8-small.

## Convert the pretrained yolov8.pt to yolov8.onnx
