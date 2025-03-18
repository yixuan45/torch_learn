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
# print(imgs.shape)

# 用切片imgs[0]取出此批次中的第一张图片
# print(imgs[0].shape)

# permute方法设置图片channel为最后一个维度
# 并使用.numpy()方法将张量转换为ndarray
# im = imgs[0].permute(1, 2, 0).numpy()
# print(im.max(), im.min())  # 打印图片的取值范围，类似(0.3803922,-0.96862745)
# im = (im + 1) / 2  # 将取值范围还原回(0,1)
# # 下面绘制图片
# plt.title(labels[0].item())
# plt.imshow(im)
# plt.show()

id_to_class = dict((v, k) for k, v in train_ds.class_to_idx.items())
# 字典推导式
# print(id_to_class)

plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs[:6], labels[:6])):
    img = (img.permute(1, 2, 0).numpy() + 1) / 2
    plt.subplot(2, 3, i + 1)
    plt.title(id_to_class.get(label.item()))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
plt.show()
