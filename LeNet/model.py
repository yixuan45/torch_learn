import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__(self)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.tanh(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.tanh(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.tanh(x)
        x = x.view(x.size(0), -1)  # 将batch展平
        # 接下来全连接
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        return x
