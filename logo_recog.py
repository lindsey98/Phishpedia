from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np
import torch


def pred_rcnn(im, predictor):
    '''
    Perform logo detection inference with the Faster R-CNN model.
    :param im: path to the screenshot image
    :param predictor: detectron2 DefaultPredictor
    :return: Nx4 tensor of logo bounding boxes, or None if the image cannot be read
    '''
    image_path = im
    im = cv2.imread(image_path)

    if im is None:
        print(f"Image at path {image_path} could not be read")
        return None

    if im.shape[-1] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

    outputs = predictor(im)

    instances = outputs['instances']
    pred_classes = instances.pred_classes  # tensor
    pred_boxes = instances.pred_boxes  # Boxes object

    logo_boxes = pred_boxes[pred_classes == 1].tensor

    return logo_boxes


def config_rcnn(cfg_path, weights_path, conf_threshold):
    '''
    Configure weights and confidence threshold
    :param cfg_path:
    :param weights_path:
    :param conf_threshold:
    :return:
    '''
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    # uncomment if you installed detectron2 cpu version
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'

    # Initialize model
    predictor = DefaultPredictor(cfg)
    return predictor


COLORS = {
    0: (255, 255, 0),  # logo
    1: (36, 255, 12),  # input
    2: (0, 255, 255),  # button
    3: (0, 0, 255),  # label
    4: (255, 0, 0)  # block
}


def vis(img_path, pred_boxes):
    '''
    Visualize rcnn predictions
    :param img_path: str
    :param pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :param pred_classes: torch.Tensor of shape Nx1 0 for logo, 1 for input, 2 for button, 3 for label(text near input), 4 for block
    :return None
    '''

    check = cv2.imread(img_path)
    if pred_boxes is None or len(pred_boxes) == 0:
        print("Pred_boxes is None or the length of pred_boxes is 0")
        return check
    pred_boxes = pred_boxes.numpy() if not isinstance(pred_boxes, np.ndarray) else pred_boxes

    # draw rectangle
    for j, box in enumerate(pred_boxes):
        if j == 0:
            cv2.rectangle(check, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLORS[0], 2)
        else:
            cv2.rectangle(check, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLORS[1], 2)

    return check
