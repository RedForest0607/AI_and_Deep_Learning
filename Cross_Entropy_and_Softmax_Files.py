import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)

z = torch.FloatTensor([1, 2, 3])
print(z)
hypothesis = F.softmax(z, dim=0)
print(hypothesis)

from numpy import exp
exp(1)

print(exp(1)/(exp(1)+exp(2)+exp(3)))
print(exp(2)/(exp(1)+exp(2)+exp(3)))
print(exp(3)/(exp(1)+exp(2)+exp(3)))

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)

print(hypothesis)
print(0.2222 + 0.2917 + 0.1478 + 0.1544 + 0.1840)

y = torch.randint(5, (3,)).long()

print(y)

y_one_hot = torch.zeros_like(hypothesis)

print(y_one_hot)
print(y_one_hot.scatter_(1, y.unsqueeze(1), 1))

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

print(cost)
print(torch.log(F.softmax(z, dim=1)))
print(F.log_softmax(z, dim=1))
print((y_one_hot * - torch.log(F.softmax(z, dim=1))).sum(dim=1).mean())
print(F.nll_loss(F.log_softmax(z, dim=1), y))
print(y)
print(F.cross_entropy(z, y))
