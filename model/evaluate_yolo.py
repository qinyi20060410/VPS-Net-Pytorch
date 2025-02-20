from __future__ import division

from net.yolo import *
from util.utils import *
from util.datasets import *
from util.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size,
             batch_size):
    # Get dataloader
    dataset = PS_Dataset(path,
                         img_size=img_size,
                         augment=False,
                         multiscale=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             collate_fn=dataset.collate_fn)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available(
    ) else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    model.eval()
    for batch_i, (imgs, targets) in enumerate(dataloader):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs,
                                          conf_thres=conf_thres,
                                          nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs,
                                               targets,
                                               iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))
    ]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives,
                                                       pred_scores,
                                                       pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="size of each image batch")
    parser.add_argument("--model_def",
                        type=str,
                        default="config/yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument("--data_config",
                        type=str,
                        default="config/custom.data",
                        help="path to data config file")
    parser.add_argument("--weights_path",
                        type=str,
                        default="weights/yolov3.pth",
                        help="path to weights file")
    parser.add_argument("--class_path",
                        type=str,
                        default="data/angle_type.names",
                        help="path to class label file")
    parser.add_argument("--iou_thres",
                        type=float,
                        default=0.5,
                        help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres",
                        type=float,
                        default=0.001,
                        help="object confidence threshold")
    parser.add_argument("--nms_thres",
                        type=float,
                        default=0.5,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size",
                        type=int,
                        default=416,
                        help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    test_path = data_config["test"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=test_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
