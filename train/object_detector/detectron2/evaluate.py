from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def evaluate(gt_coco_path, results_coco_path):
    coco_gt = COCO(gt_coco_path)
    coco_dt = coco_gt.loadRes(results_coco_path)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-json",
        required=True,
        help="Path to ground-truth bbox",
    )

    parser.add_argument(
        "--pred-json",
        required=True,
        help="Path to predicted bbox",
    )

    args = parser.parse_args()

    evaluate(args.gt_json, args.pred_json)
