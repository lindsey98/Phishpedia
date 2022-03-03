"""Contains functions to visualize pre-processed training data, and also
compare ground-truth to prediction results.

See:
- detectron2/tools/visualize_data.py
- detectron2/tools/visualize_json_results.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from random import sample
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import wandb
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.visualizer import Visualizer
from fvcore.common.file_io import PathManager
from PIL import Image
from tqdm import tqdm

from src.detectron2_pedia.detectron2_1.datasets import BenignMapper


def viz_data(cfg) -> List[wandb.Image]:
    """Returns a sample of the training image examples along with its annotations.

    Parameters
    ----------
    cfg :

    Returns
    -------
    List[wandb.Image]
    """
    scale = 0.5

    # Grab train dataset and its metadata
    train_data_loader = build_detection_train_loader(
        cfg, mapper=BenignMapper(cfg, is_train=True)
    )
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    batch = next(iter(train_data_loader))
    imgs = []

    # Sample only the first 8 images and its annotations
    for i in range(8):
        per_image = batch[i]

        img = per_image["image"].permute(1, 2, 0)
        if cfg.INPUT.FORMAT == "BGR":
            img = img[:, :, [2, 1, 0]]
        else:
            img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

        visualizer = Visualizer(img, metadata=metadata, scale=scale)
        target_fields = per_image["instances"].get_fields()
        labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
        vis = visualizer.overlay_instances(
            labels=labels,
            boxes=target_fields.get("gt_boxes", None),
            masks=target_fields.get("gt_masks", None),
            keypoints=target_fields.get("gt_keypoints", None),
        )

        # For wandb logging
        imgs.append(wandb.Image(vis.get_image()))

    return imgs

    # Plot and return images in a grid
    # return plot_imgs(imgs, n_cols=2)


def viz_preds(cfg) -> List[wandb.Image]:
    """Returns a sample of image predictions and its corresponding groundtruth.

    Parameters
    ----------
    cfg :

    Returns
    -------
    List[wandb.Image]
    """
    output_path = Path(cfg.OUTPUT_DIR)
    # Requires JSON predictions file
    predictions_path = output_path / "coco_instances_results.json"
    val_dataset = cfg.DATASETS.TEST[0]
    # To filter out predictions
    conf_threshold = 0.1

    # Load predictions JSON file
    with PathManager.open(str(predictions_path), "r") as f:
        # List of instance predictions
        predictions = json.load(f)

    # Group predictions for each image
    # i.e. image_id -> List[predictions]
    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    # Get groundtruth annotations
    dicts = list(DatasetCatalog.get(val_dataset))
    metadata = MetadataCatalog.get(val_dataset)

    # Sample images to visualize
    imgs = []
    n_imgs = 8
    dicts = sample(dicts, n_imgs)

    for dic in tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]

        # Creates Instances object
        predictions = pred_by_image[dic["image_id"]]
        predictions = create_instances(
            predictions, img.shape[:2], metadata, conf_threshold
        )

        # Draw instance-level predictions on an image
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        # Draw ground-truth annotations on an image
        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        # Place them side by side
        concat = np.concatenate((vis_pred, vis_gt), axis=1)

        # For wandb logging
        imgs.append(wandb.Image(concat))

    return imgs

    # Plot and return images
    # return plot_imgs(imgs, n_cols=1, size=(10, 20))


# HELPER FUNCTIONS #############################################################
def plot_imgs(
    imgs: List[np.array], n_cols: int, size: Tuple[int, int] = (10, 10)
) -> plt.figure:
    """Plots a list of images in a grid.

    Parameters
    ----------
    imgs : List[np.array]
        List of images in np.array format
    n_cols : int
        Number of columns for the grid
    size : Tuple[int, int]
        Size of each image, i.e. (H, W), by default (10, 10)

    Returns
    -------
    plt.figure
    """
    fig = plt.gcf()

    n_rows = math.ceil(len(imgs) / n_cols)

    height, width = size
    fig.set_size_inches(n_cols * width, n_rows * height)

    for i, img in enumerate(imgs):
        sp = plt.subplot(n_rows, n_cols, i + 1)
        sp.axis("Off")

        plt.imshow(img)

    # return fig
    return plt


def create_instances(predictions, image_size, metadata, conf_threshold) -> Instances:
    """Create the Instances object from a list of instance prediction dicts
    from a single image.
    """

    def dataset_id_map(ds_id):
        return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    ret = Instances(image_size)

    # Filter instances by confidence threshold
    score = np.asarray([x["score"] for x in predictions])
    # Get indices of chosen instances
    chosen = (score > conf_threshold).nonzero()[0]
    score = score[chosen]

    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    # See if segmentation predictions are available
    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass

    return ret
