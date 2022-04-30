import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
sigmoid = torch.nn.Sigmoid()
criterion = torch.nn.BCELoss().to(device)

# 다층 퍼셉트론 MLP
# 입력층 2, 은닉층 2, 은닉층2, 출력층1
linear1 = torch.nn.Linear(2, 2, bias=True)
linear2 = torch.nn.Linear(2, 1, bias=True)
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, cost.item())
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('\nCase: \nLearning Rate: 0.01\nHypothesis: ', hypothesis.detach(), '\nCorrect: ', predicted.detach(), '\nAccuracy: ', accuracy.item())