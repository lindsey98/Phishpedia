# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
# !/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import numpy as np
import torch
import torchvision as tv

import train.siamese.lbtoolbox as lb
import train.siamese.mobilenet as models
import train.siamese.bit_common as bit_common
import train.siamese.bit_hyperrule as bit_hyperrule
from train.siamese.data import GetLoader

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i


def mktrainval(args, logger):
    """Returns train and validation datasets."""
    precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
    train_tx = tv.transforms.Compose([
        tv.transforms.Resize((precrop, precrop)),
        tv.transforms.RandomCrop((crop, crop)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.dataset == "logo_2k":
        train_set = GetLoader(data_root='./datasets/siamese_training/Logo-2K+',
                              data_list='./datasets/siamese_training/List/train_images_root.txt',
                              label_dict='./datasets/siamese_training/List/logo2k_labeldict.pkl',
                              transform=train_tx)

        valid_set = GetLoader(data_root='./datasets/siamese_training/Logo-2K+',
                              data_list='./datasets/siamese_training/List/test_images_root.txt',
                              label_dict='./datasets/siamese_training/List/logo2k_labeldict.pkl',
                              transform=val_tx)

    elif args.dataset == "targetlist":
        train_set = GetLoader(data_root='./models/expand_targetlist',
                              data_list='./datasets/siamese_training/train_targets.txt',
                              label_dict='./datasets/siamese_training/target_dict.pkl',
                              transform=train_tx)

        valid_set = GetLoader(data_root='./models/expand_targetlist',
                              data_list='./datasets/siamese_training/test_targets.txt',
                              label_dict='./datasets/siamese_training/target_dict.pkl',
                              transform=val_tx)

    logger.info("Using a training set with {} images.".format(len(train_set)))
    logger.info("Using a validation set with {} images.".format(len(valid_set)))
    logger.info("Num of classes: {}".format(len(valid_set.classes)))

    micro_batch_size = args.batch // args.batch_split

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=micro_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    if micro_batch_size <= len(train_set):
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False)
    else:
        # In the few-shot cases, the total dataset size might be smaller than the batch-size.
        # In these cases, the default sampler doesn't repeat, so we need to make it do that
        # if we want to match the behaviour from the paper.
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

    return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, device, chrono, logger, step):
    # switch to evaluate mode
    model.eval()

    logger.info("Running validation...")
    logger.flush()

    all_c, all_top1, all_top5 = [], [], []
    end = time.time()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # measure data loading time
            chrono._done("eval load", time.time() - end)

            # compute output, measure accuracy and record loss.
            with chrono.measure("eval fprop"):
                logits, _ = model(x)
                c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                top1, top5 = topk(logits, y, ks=(1, 5))
                all_c.extend(c.cpu())  # Also ensures a sync point.
                all_top1.extend(top1.cpu())
                all_top5.extend(top5.cpu())

        # measure elapsed time
        end = time.time()

    model.train()
    logger.info("Validation@{} loss {:.5f}, ".format(step, np.mean(all_c)))
    logger.info("top1 {:.2%}, ".format(np.mean(all_top1)))
    logger.info("top5 {:.2%}".format(np.mean(all_top5)))
    logger.flush()
    return np.mean(all_c), all_top1, all_top5


def mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    indices = torch.randperm(x.shape[0]).to(x.device)

    mixed_x = l * x + (1 - l) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def main(args):
    logger = bit_common.setup_logger(args)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Going to train on {}".format(device))

    train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)

    model = models.__all__[args.model](pretrained=True, num_classes=len(valid_set.classes))

    logger.info("Moving model onto all GPUs")
    model = torch.nn.DataParallel(model)

    # Note: no weight-decay!
    optim = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)

    # Optionally resume from a checkpoint.
    # Load it to CPU first as we'll move the model to GPU later.
    # This way, we save a little bit of GPU memory when loading.
    step = 0

    # If pretrained weights are specified
    if args.weights_path:
        logger.info("Loading weights from {}".format(args.weights_path))
        checkpoint = torch.load(args.weights_path, map_location="cpu")
        # New task might have different classes; remove the pretrained classifier weights
        del checkpoint['model']['module.classifier.1.weight']
        del checkpoint['model']['module.classifier.1.bias']
        model.load_state_dict(checkpoint["model"], strict=False)

    # Resume fine-tuning if we find a saved model.
    savename = pjoin(args.logdir, args.name, "bit.pth.tar")
    try:
        logger.info("Model will be saved in '{}'".format(savename))
        checkpoint = torch.load(savename, map_location="cpu")
        logger.info("Found saved model to resume from at '{}'".format(savename))

        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        logger.info("Resumed at step {}".format(step))

    except FileNotFoundError:
        logger.info("Fine-tuning from BiT")

    # Send to GPU
    model = model.to(device)
    optimizer_to(optim, device)
    optim.zero_grad()

    model.train()
    mixup = bit_hyperrule.get_mixup(len(train_set))
    cri = torch.nn.CrossEntropyLoss().to(device)

    logger.info("Starting training!")
    chrono = lb.Chrono()

    mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
    end = time.time()
    best_loss = float('inf')  # Initialize the best loss with infinity

    with lb.Uninterrupt() as u:
        for x, y in recycle(train_loader):
            # measure data loading time, which is spent in the `for` statement.
            chrono._done("load", time.time() - end)

            if u.interrupted:
                break

            # Schedule sending to GPU(s)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Update learning-rate, including stop training if over.
            lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
            if lr is None:
                break
            for param_group in optim.param_groups:
                param_group["lr"] = lr

            if mixup > 0.0:
                x, y_a, y_b = mixup_data(x, y, mixup_l)

            # compute output
            logits, _ = model(x)
            if mixup > 0.0:
                c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
            else:
                c = cri(logits, y)
            c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

            # Accumulate grads
            (c / args.batch_split).backward()

            logger.info("[step {}]: loss={:.5f} (lr={:.1e})".format(step, c_num, lr))
            logger.flush()

            # Update params
            optim.step()
            optim.zero_grad()
            step += 1

            # Sample new mixup ratio for next batch
            mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

            # Save model
            end = time.time()
            if step % 50 == 0:
                val_loss, _, _ = run_eval(model, valid_loader, device, chrono, logger, step=step)
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({
                        "step": step,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "best_loss": best_loss,
                    }, savename)
                    logger.info(f"New best model saved at step {step} with validation loss {val_loss:.5f}")

        # Final eval at end of training.
        run_eval(model, valid_loader, device, chrono, logger, step='end')
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
        }, savename)

    logger.info("Timings:\n{}".format(chrono))


if __name__ == "__main__":
    parser = bit_common.argparser(models.__all__)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    main(parser.parse_args())


## image size of 32: top1 71.80%, top5 86.51%

## image size of 64: top1 90.37%, top5 96.15%

## image size of 128, Top-1 Accuracy: 94.05% Top-5 Accuracy: 97.90%
