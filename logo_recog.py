import argparse
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np
import time
import torch

def pred_rcnn(im, predictor, return_predictor_runtime=False):
    '''
    Perform inference for RCNN
    :param im:
    :param predictor:
    :return:
    '''
    im = cv2.imread(im)

    if im is not None:
        if im.shape[-1] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    else:
        return None, None, None, None

    start_time = time.time()
    outputs = predictor(im)
    end_time = time.time()
    predictor_runtime = end_time - start_time

    instances = outputs['instances']
    pred_classes = instances.pred_classes  # tensor
    pred_boxes = instances.pred_boxes  # Boxes object

    logo_boxes = pred_boxes[pred_classes == 1].tensor
    input_boxes = pred_boxes[pred_classes == 0].tensor

    scores = instances.scores  # tensor
    logo_scores = scores[pred_classes == 1]
    input_scores = scores[pred_classes == 0]

    if return_predictor_runtime:
        ret = (logo_boxes, logo_scores, input_boxes, input_scores, predictor_runtime)
    else:
        ret = (logo_boxes, logo_scores, input_boxes, input_scores)
    return ret


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
        return check
    pred_boxes = pred_boxes.numpy() if not isinstance(pred_boxes, np.ndarray) else pred_boxes

    # draw rectangle
    for j, box in enumerate(pred_boxes):
        if j == 0:
            cv2.rectangle(check, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
        else:
            cv2.rectangle(check, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (36, 255, 12), 2)

    return check


def main():
    parser = argparse.ArgumentParser(description='Logo Recognition')
    parser.add_argument('-i', '--input', help='Input image file path', required=True)
    parser.add_argument('-o', '--output', help='Output image file path', default='output.png')
    args = parser.parse_args()
    input_img_path = args.input
    output_img_path = args.output

    cfg_path = 'models/faster_rcnn.yaml'
    weights_path = 'models/rcnn_bet365.pth'
    conf_threshold = 0.05

    predictor = config_rcnn(cfg_path, weights_path, conf_threshold)
    logo_boxes, logo_scores, input_boxes, input_scores, predictor_runtime = pred_rcnn(input_img_path, predictor, return_predictor_runtime=True)
    print(f"Predictor runtime: {predictor_runtime:.3g} seconds")
    plotvis = vis(output_img_path, logo_boxes)
    cv2.imwrite(output_img_path, plotvis)
    print(f"Saved output to {output_img_path}")

if __name__ == '__main__':
    main()