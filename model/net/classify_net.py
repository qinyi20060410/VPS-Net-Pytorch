#! ~/.miniconda3/envs/pytorch/bin/python

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classify_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Classify_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=(3, 9), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(40, 80, kernel_size=(3, 5), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 0)),
            nn.Conv2d(80, 120, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(120, 160, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(160 * 5 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 160 * 5 * 5)
        x = self.classifier(x)
        return x
