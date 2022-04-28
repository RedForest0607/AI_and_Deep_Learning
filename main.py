# 파이토치 실행 샘플
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(2)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Linear regresstion의 x축과 y축
x_train = torch.FloatTensor([[1], [2], [3]])
# 토치로 선언, 플롯텐서로
y_train = torch.FloatTensor([[1], [2], [3]])

print(x_train)
print(x_train.shape)
# 3by1
W=torch.zeros(1, requires_grad=True)
print(W)
# 0으로 생성

b = torch.zeros(1, requires_grad=True)

#가설 선언
hypothesis = x_train*W+b
print(hypothesis)
print((hypothesis-y_train)**2)
cost = torch.mean((hypothesis-y_train)**2)
print(cost)
optimizer = optim.SGD([W, b], lr=0.01)
# 같이 다니는 세줄
#미분값을 0으로 초기화 해줌
optimizer.zero_grad()
cost.backward()
# 계산 방향
optimizer.step()
# 계산 시작
print(W)
print(b)

hypothesis = x_train*W+b
print('hypothesis: ', hypothesis)

cost = torch.mean((hypothesis-y_train)**2)

