# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision.transforms import ToTensor

train_ds = torchvision.datasets.MNIST('data/', train=True, transform=ToTensor(), download=True)
test_ds = torchvision.datasets.MNIST('data/', train=False, transform=ToTensor(), download=True)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64)
img, labels = next(iter(train_dl))
print(img)
print(labels)
