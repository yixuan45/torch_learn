# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

train_dir = r'2_class/train'
test_dir = r'2_class/test'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_ds = torchvision.datasets.ImageFolder(test_dir, transform=transform)

# print(train_ds.classes)  # 打印次dataset的类别:['airplane','lake']
# print(train_ds.class_to_idx)  # 输出类别编码{'airplane':0,'lake':1}
# print(len(train_ds), len(tes_ds))  # 打印两个dataset的大小，输出(1120,280)

BATCHSIZE = 16
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCHSIZE)
imgs, labels = next(iter(train_dl))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 30 * 30, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
model = Net().to(device)
preds = model(imgs.to(device))

print(imgs.shape)
print(preds.shape)
print(torch.argmax(preds, 1))
