import torch
import torch.nn as nn

m = nn.Linear(2, 3)
input = torch.randn(5, 2)
print(input)
output = m(input)
print(output)
print(output.size())
