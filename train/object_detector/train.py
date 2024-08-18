import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
from ultralytics import YOLO
import argparse

def main(args):
    # Load a model
    model = YOLO(args.base_model)

    # Train the model
    results = model.train(
        name=args.name,
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        device=args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model with specified parameters.")

    # Adding arguments
    parser.add_argument("--base_model", type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument("--name", type=str, required=True, help="Name of the training run.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the dataset configuration file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training. Default is 320.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train. Default is 20.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training. Default is 16.")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate. Default is 1e-4.")
    parser.add_argument("--lrf", type=float, default=1e-4, help="Final learning rate. Default is 1e-4.")
    parser.add_argument("--device", type=int, nargs='+', default=[0, 2, 1, 3],
                        help="List of device IDs to use. Default is [0, 2, 1, 3].")

    args = parser.parse_args()
    main(args)

### For yolo nano
# I have tried size = 128 but get underfitting problem
# size = 320
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
#                    all       1561       3744      0.815      0.776      0.813      0.526

# size = 640
#           Class    Images  Instances       Box(P          R            mAP50         mAP50-95):
#           all      1561       3744         0.82947,      0.81471,     0.85239,        0.59849

### For yolo small
# Class    Images  Instances       Box(P          R            mAP50         mAP50-95):
# all       1561       3744      0.856      0.862       0.89      0.634

###



