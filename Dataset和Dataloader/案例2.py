# -*- coding: utf-8 -*-
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        filename = "./anli2/data.xlsx"
        data = pd.read_excel(filename)
        self.x1 = data['x1']
        self.x2 = data['x2']
        self.x3 = data['x3']
        self.x4 = data['x4']
        self.y = data['y']

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):
        return self.x1[item], self.x2[item], self.x3[item], self.x4[item], self.y[item]


if __name__ == '__main__':
    mydataset = MyDataset()
    mydataloader = DataLoader(mydataset, shuffle=True, batch_size=4)
    for x1, x2, x3, x4, y in mydataloader:
        print(f"x1={x1},x2={x2},x3={x3},x4={x4},y={y}")
