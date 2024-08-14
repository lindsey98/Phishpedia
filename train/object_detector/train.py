import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
from ultralytics import YOLO

# Load a model
model = YOLO(
    "yolov8n.pt"
)  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    name="rcnn_nano",
    data="./datasets/object_detector_training/config.yaml",
    epochs=10,
    batch=16,
    lr0=1e-4,
    lrf=1e-4,
    device=[0, 2, 1, 3],
)





