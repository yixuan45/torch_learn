# -*- coding: utf-8 -*-
import os
import cv2 as cv
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class MyImageDataset(Dataset):
    def __init__(self):
        image_root = r"anli3/image"
        self.file_path_list = []
        dir_name = []
        self.labels = []

        for root, dirs, files in os.walk(image_root):
            if dirs:
                dir_name = dirs
            for file_i in files:
                file_i_full_path = os.path.join(root, file_i)
                self.file_path_list.append(file_i_full_path)
                label = root.split(os.sep)[-1]
                self.labels.append(label)

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, item):
        img = cv.imread(self.file_path_list[item])
        img = cv.resize(img, dsize=(256, 256))
        # 原先的shape为[1,256,256,3]
        # 要将3调换到1的后面
        img = np.transpose(img, (2, 1, 0))
        img_tensor = torch.from_numpy(img)
        label = self.labels[item]
        return img_tensor, label


if __name__ == '__main__':
    mydataset = MyImageDataset()
    mydataloader = DataLoader(mydataset, batch_size=4, shuffle=True, num_workers=4)
    for x_i, y_i in mydataloader:
        print(x_i.shape, y_i)
