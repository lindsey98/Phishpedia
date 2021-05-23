#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import detectron2.utils.comm as comm
import yaml
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA

from src.detectron2_pedia.detectron2_1.datasets import BenignMapper
from src.detectron2_pedia.detectron2_1.register_backbone import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# Implement evaluation here
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        return COCOEvaluator(dataset_name, cfg, False, cfg.OUTPUT_DIR)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    # Insert custom data loading logic
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=BenignMapper(cfg, is_train=True)
        )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    # Can create custom configs fields here too
    cfg.merge_from_list(args.opts)

    # Create directory if does not exist
    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def main(args):
    cfg = setup(args)

    # Load cfg as python dict
    config = load_yaml(args.config_file)

    # If evaluation
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)

    # If training
    else:
        trainer = Trainer(cfg)
        # Load model weights (if specified)
        trainer.resume_or_load(resume=args.resume)
        if cfg.TEST.AUG.ENABLED:
            trainer.register_hooks(
                [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
            )
        # Will evaluation be done at end of training?
        res = trainer.train()

    return res


if __name__ == "__main__":
    # Create a parser with some common arguments
    parser = default_argument_parser()
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
