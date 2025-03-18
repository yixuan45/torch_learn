# -*- coding: utf-8 -*-
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.liner_1 = nn.Linear(16 * 4 * 4, 256)
        self.liner_2 = nn.Linear(256, 10)

    def forward(self, input):
        x = F.max_pool2d(F.relu(input), 2)
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.liner_1(x))
        x = self.liner_2(x)
        return x

