#! ~/.miniconda3/envs/pytorch/bin/python
from __future__ import division

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from net.classify_net import *
from util.logger import *
from util.utils import *
from util.datasets import *
from util.parse_config import *
from evaluate_yolo import evaluate

from terminaltables import AsciiTable

import time
import datetime
import argparse

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
    parser.add_argument("--data_config",
                        type=str,
                        default="config/custom.data",
                        help="path to data config file")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    test_path = data_config["test"]
    class_names = load_classes(data_config["names"])

    model = Classify_Net().to(device)
    model.apply(weights_init_normal)

    # if opt.pretrained_weights:
    #     if opt.pretrained_weights.endswith(".pth"):
    #         model.load_state_dict(torch.load(opt.pretrained_weights))
    #     else:
    #         model.load_darknet_weights(opt.pretrained_weights)

    train_dataset = PS_Dataset_C(train_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=30,
                                             gamma=0.1)
    loss_fun = torch.nn.MSELoss()

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()

        for batch_i, (imgs, targets) in enumerate(train_dataloader):

            print('i', batch_i)
            batches_done = len(train_dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device))

            outputs = model(imgs)

            # print(outputs)
            print(targets)

            loss = loss_fun(outputs, targets)

            print(loss)
            print(outputs)