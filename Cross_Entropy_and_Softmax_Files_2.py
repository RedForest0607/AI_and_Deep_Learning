import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)

z = torch.FloatTensor([1, 2, 3])
x = torch.FloatTensor([[1],[2],[3]])
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
hypothesis = F.softmax(x, dim=0)
print(hypothesis)
