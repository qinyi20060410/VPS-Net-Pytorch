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
import numpy as np
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
                        default=64,
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

    test_dataset = PS_Dataset_C(test_path)
    test_dataloader = torch.utils.data.DataLoader(
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
    loss_fun = torch.nn.CrossEntropyLoss()

    total_train_steps = 1
    total_test_steps = 1
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []

    ap = -1

    for epoch in range(opt.epochs):
        lr_scheduler.step()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num = 0
        for batch_i, (imgs, targets) in enumerate(train_dataloader):
            if imgs is not None and targets is not None:
                batches_done = len(train_dataloader) * epoch + batch_i

                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device))
                targets = torch.reshape(targets, (-1, ))

                outputs = model(imgs)
                outputs = torch.reshape(outputs, (-1, 3))
                loss = loss_fun(outputs, targets)
                loss.backward()

                if batches_done % opt.gradient_accumulations:
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct = torch.sum(preds == targets).item()
                num += correct
                total += targets.size(0)

                if total_train_steps % 100 == 0:
                    print(
                        'Epoch: {} \tStep: {} \tLoss: {:.6f} \tCorrect: {} \tAll:{}'
                        .format(epoch + 1, total_train_steps, loss.item(),
                                correct, targets.size(0)))

                total_train_steps += 1

        train_acc.append(num / total)
        train_loss.append(running_loss / len(train_dataloader))
        print('train loss: {:.6f}, train acc: {:.6f}'.format(
            np.mean(train_loss), (num / total)))

        running_loss = 0.0
        correct = 0
        total = 0
        num = 0
        model.eval()
        with torch.no_grad():
            for batch_i, (imgs, targets) in enumerate(test_dataloader):
                if imgs is not None and targets is not None:
                    batches_done = len(train_dataloader) * epoch + batch_i

                    imgs = Variable(imgs.to(device))
                    targets = Variable(targets.to(device))
                    targets = torch.reshape(targets, (-1, ))

                    outputs = model(imgs)
                    outputs = torch.reshape(outputs, (-1, 3))
                    loss = loss_fun(outputs, targets)

                    running_loss += loss.item()
                    _, preds = torch.max(outputs.data, 1)
                    correct = torch.sum(preds == targets).item()
                    num += correct
                    total += targets.size(0)

                    if total_test_steps % 100 == 0:
                        print(
                            'Epoch: {} \tStep: {} \tLoss: {:.6f} \tCorrect: {} \tAll:{}'
                            .format(epoch + 1, total_test_steps, loss.item(),
                                    correct, targets.size(0)))
                    total_test_steps += 1

        val_acc.append(num / total)
        val_loss.append(running_loss / len(train_dataloader))
        print('test loss: {:.6f}, test acc: {:.6f}'.format(
            np.mean(train_loss), (num / total)))

        if (num / total > ap):
            print(f"save best model params")
            ap = num / total
            torch.save(model.state_dict(),
                       f"checkpoints/classify_ckpt_best.pth")
