import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 파이토치에 내장 되어 있는 선형 회귀 함수
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# 모델 초기화
model = LinearRegressionModel()
# optimzer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs +1):
    # H(x) 계산
    prediction = model(x_train)
    # cost 계산
    # 평균 제곱 오차 함수 nn.functional.mse_loss()
    cost = F.mse_loss(prediction, y_train)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    # 파이토치는 미분을 통해서 얻은 기울기를 누적시키기 때문에 optimizer.zero_grad()를 통해 초기화 시켜야 한다.
    cost.backward()
    optimizer.step()
    # 100번 마다 로그 출력
    if epoch % 100 == 0:
        params = list(model.parameters())
        W = params[0].item()
        b = params[1].item()
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs, W, b, cost.item()))
