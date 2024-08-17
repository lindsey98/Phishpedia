from memory_profiler import profile
from ultralytics import YOLO
import cv2
import torch

@torch.inference_mode()
def logo_recog(object_detector, screenshot_path, imgsz):
    if isinstance(object_detector, YOLO):
        pred_results = object_detector.predict([screenshot_path],
                                              device="cpu",
                                              classes=[1], # only keep logo class
                                              save=False,
                                              save_txt=False,
                                              save_conf=False,
                                              imgsz=imgsz, # input image size
                                              verbose=False,
                                                  )
        pred_boxes = pred_results[0].boxes.xyxy.detach().cpu().numpy()
    else:
            im = cv2.imread(screenshot_path)
            if im is not None:
                if im.shape[-1] == 4:
                    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
            else:
                return None

            outputs = object_detector(im)

            instances = outputs['instances']
            pred_classes = instances.pred_classes  # tensor
            pred_boxes = instances.pred_boxes  # Boxes object

            pred_boxes = pred_boxes[pred_classes == 1].tensor.detach().cpu().numpy()

    return pred_boxes

def config_rcnn(cfg_path, weights_path, conf_threshold):
    '''
    Configure weights and confidence threshold
    :param cfg_path:
    :param weights_path:
    :param conf_threshold:
    :return:
    '''
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    # uncomment if you installed detectron2 cpu version
    cfg.MODEL.DEVICE = 'cpu'

    # Initialize model
    predictor = DefaultPredictor(cfg)
    return predictor