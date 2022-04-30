import torch
import torch.nn as nn

conv = nn.Conv2d(1, 1, 11, stride=4, padding=0)
input = torch.Tensor(1, 1, 227, 227)
input.shape
out = conv(input)
out.shape
conv2 = nn.Conv2d(1, 1, 7, stride=2, padding=0)
input2 = torch.Tensor(1, 1, 64, 64)
conv2(input2)
out2 = conv2(input)
out2.shape
conv3 = nn.Conv2d(1, 1, 5, stride=1, padding=2)
conv3
input3 = torch.Tensor(1, 1, 32, 32)
out3 = conv3(input3)
out3.shape
conv4 = nn.Conv2d(1, 1, 5)
input4 = torch.Tensor(1, 1,32, 64)
conv4
input4.shape
out4 = conv4(input4)
out4.shape
conv5 = nn.Conv2d(1, 1, 3, padding=1)
input5 = torch.Tensor(1, 1, 64, 32)
out5 = conv5(input5)
out5.shape

input = torch.Tensor(1, 1, 28, 28)
conv1 = nn.Conv2d(1, 5, 5)
pool = nn.MaxPool2d(2)
out = conv1(input)
out2 = pool(out)
out.size()
out2.size()
