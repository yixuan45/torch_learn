# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader


class NewDataset(Dataset):
    def __init__(self):
        self.x = [i for i in range(12)]
        self.y = [i * 2 for i in range(12)]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    newdataset = NewDataset()
    newdataloader = DataLoader(newdataset)
    for x_i, y_i in newdataloader:
        print(x_i, y_i)
    newdataloader = DataLoader(newdataset, batch_size=2)
    for x_i, y_i in newdataloader:
        print(x_i, y_i)
    newdataloader = DataLoader(newdataset, batch_size=4, shuffle=True)
    for x_i, y_i in newdataloader:
        print(x_i, y_i)
