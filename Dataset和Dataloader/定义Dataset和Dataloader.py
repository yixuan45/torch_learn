# -*- coding: utf-8 -*-
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        # 定义了数据集包含了什么东西
        self.x = []
        self.y = []

    def __len__(self):
        # 返回数据集的总长度
        return len(...)

    def __getitem__(self, idx):
        # 当数据集被读取时，需要返回的数据
        ...
        return self.x[idx], self.y[idx]
