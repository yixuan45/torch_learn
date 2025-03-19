# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import glob
from torchvision import transforms
from torch.utils import data
from PIL import Image

imgs = glob.glob(r'./dataset2/*.jpg')
print(imgs[:3])

species = ['cloudy', 'rain', 'shine', 'sunrise']  # 4种类别名称

# 字典推导式获取类别到编号的字典
species_to_idx = dict((c, i) for i, c in enumerate(species))
print(species_to_idx)

# 字典推到是获取编号到类别的字典
idx_to_species = dict((v, k) for k, v in species_to_idx.items())
print(idx_to_species)

labels = []
for img in imgs:
    for i, c in enumerate(species):
        if c in img:
            labels.append(i)
print(labels[:3])

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


# 创建Dataset类，这里自定义的Dataset类名为WT_dataset,创建Dataset类需要继承data.Dataset这个父类，同时重写__getitem__()方法和__len__()方法
class WT_dataset(data.Dataset):
    def __init__(self, imgs_path, labels):
        self.imgs_path = imgs_path
        self.labels = labels

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = self.labels[index]

        pil_img = Image.open(img_path)
        pil_img = pil_img.convert("RGB")  # 此行可选，如有黑白图片会被转为RGB格式
        pil_img = transform(pil_img)
        return pil_img, label

    def __len__(self):
        return len(self.imgs_path)



