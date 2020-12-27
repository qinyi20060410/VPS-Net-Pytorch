#! ~/.miniconda3/envs/pytorch/bin/python
from __future__ import division

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from net.yolo import *
from util.logger import *
from util.utils import *
from util.datasets import *
from util.parse_config import *
from evaluate_yolo import evaluate

from terminaltables import AsciiTable

import time
import datetime
import argparse

import onnx
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",
                        type=int,
                        default=500,
                        help="number of epochs")
    parser.add_argument("--batch_size",
                        type=int,
                        default=6,
                        help="size of each image batch")
    parser.add_argument("--gradient_accumulations",
                        type=int,
                        default=2,
                        help="number of gradient accums before step")
    parser.add_argument("--model_def",
                        type=str,
                        default="config/yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument("--data_config",
                        type=str,
                        default="config/custom.data",
                        help="path to data config file")
    parser.add_argument("--pretrained_weights",
                        type=str,
                        default='weights/yolov3.weights',
                        help="if specified starts from checkpoint model")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size",
                        type=int,
                        default=416,
                        help="size of each image dimension")
    parser.add_argument("--compute_map",
                        default=False,
                        help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training",
                        default=True,
                        help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    test_path = data_config["test"]
    class_names = load_classes(data_config["names"])

    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    train_dataset = PS_Dataset(train_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn)

    optimizer = torch.optim.Adam(model.parameters())

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=30,
                                             gamma=0.1)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    best_ap = -1

    # export
    dummpy_input = torch.randn(2, 3, 416, 416, device='cuda')
    input_names = ['inputs']
    output_names = ['outputs']
    model.train(False)
    torch.onnx.export(model,
                      dummpy_input,
                      'yolo3.onnx',
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes={
                          'inputs': {
                              0: 'batch'
                          },
                          'outputs': {
                              0: 'batch'
                          }
                      },
                      opset_version=11)
    # test
    test = onnx.load('yolo3.onnx')
    onnx.checker.check_model(test)
    print("==> Passed")

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        lr_scheduler.step()
        for batch_i, (imgs, targets) in enumerate(train_dataloader):
            batches_done = len(train_dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device))
            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                optimizer.step()
                optimizer.zero_grad()

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
                epoch, opt.epochs, batch_i, len(train_dataloader))

            metric_table = [[
                "Metrics",
                *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]
            ]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [
                    formats[metric] % yolo.metrics.get(metric, 0)
                    for yolo in model.yolo_layers
                ]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(train_dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left *
                                           (time.time() - start_time) /
                                           (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        # eval
        print("\n---- Evaluating Model ----")
        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=test_path,
            iou_thres=0.5,
            conf_thres=0.5,
            nms_thres=0.5,
            img_size=opt.img_size,
            batch_size=8,
        )
        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]
        # logger.list_of_scalars_summary(evaluation_metrics, epoch)

        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")

        current_ap = AP.mean()

        # save
        if current_ap > best_ap:
            print(f"save best model params")
            best_ap = current_ap
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_best.pth")
