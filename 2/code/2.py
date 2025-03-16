# -*- coding: utf-8 -*-
from torch import nn


class EIModel(nn.Module):
    def __init__(self):
        super(EIModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, inputs):
        logits = self.linear(inputs)
        return logits



if __name__ == '__main__':
    model=EIModel()
