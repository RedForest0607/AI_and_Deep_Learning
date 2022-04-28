## Linear Regression
***
- 폴더를 자신의 학번으로 만들고 해당 폴더에서 Pytorch 설치 후 진행
- 동영상 처음 부분에 torch.manual_seed(2)로 하여 2번 진행 후 print(hypothesis)를 통해 값을 구해본다
- Linear_regression.py파일을 수정하여 y_train을 1,2,3에서 1,4,9로 했을 때 1000번을 돌린 뒤 w와 b의 값을 구해본다.
- y=x2 그래프와 w,b를 구한 linear model을 같이 그려보고 왜 저런 w와 b가 나왔는지 생각해본다. (hint: Why? Linear, Quadratic)
#
- 캡쳐할 것:
  - 1. 설치폴더 이름이 자신의 학번인 것
  - 2. seed(2)로 하여 2번 돌려본 뒤 print된 hypothesis값
  - 3. linear_regression.py파일의 y_train을 수정한 뒤 1000번 돌린 뒤의 w와 b의 값
- 보고서 작성내용:
  - 생각해본 내용 작성

#
## Preceptron & MLP
***
- XOR 문제를 똑같이 수행하여 만 번 수행 뒤에도 에러가 줄지 않음을 확인
- XOR이 아니라 책의 연습문제와 같이 (0,0) / (0,1),(1,0),(1,1) 으로 나눴을 때 어떻게 되는지 확인
- MLP 구조로 XOR을 똑같이 수행하고 learning rate 1, 0.1, 0.01 에 대해 만 번 수행하여 차이점 비교
#
- 캡쳐할 것: 
  - 1. XOR문제를 perceptron 구조에서 만 번 수행해도 에러가 줄지 않음 확인결과
  - 2. 선형분리가 가능한 (0,0) / (0,1),(1,0),(1,1) 에 대해 perceptron구조로 만 번 수행한 결과 
  - 3. MLP구조로 XOR문제를 풀 때, leraning rate를 3가지 경우에 대해 수행한 결과
보고서 작성내용:
  - learning rate에 따른 결과에 대한 고찰
#
## Convolution
***
- 동영상을 참고하여 AlexNet의 입력부터 MAX POOL1까지가 아래와 같이 volume size가 잘 나오는지 결과 확인  
  
  [227x227x3] INPUT  
  [55x55x96] CONV1: 96 11x11 filters at stride 4, pad 0  
  [27x27x96] MAX POOL1: 3x3 filters at stride 2
#
- 인셉션 모듈이 아래와 같을 때 3x3 conv, 5x5 conv, 3x3 max pooling 세가지가 모두 stride가 1일 경우
각각의 padding은 얼마이어야 그림과 같은 결과가 나오는지 생각해보고 코드를 수행한 결과 확인  
#
### 캡쳐할 것: 
  1. AlexNet CONV1, MAX POOL1 결과
  2. 인셉션 모듈 3x3 conv, 5x5, conv, 3x3 max pool 결과
  보고서 작성내용:
  없음!
#
## Cross Entropy & Softmax
***
### 동영상의 코드 그대로 진행하여 전부 캡쳐
  1. (추가) softmax를 (1,2,3)에 적용한 값과 F.softmax를 통해 구한 값 같은지 확인 
  2. 1.5608이 다 잘 나오는지 확인
  3. 파일 train_CE_low.py, train_CE_high.py 결과 비교 확인
  4. 파일 train_module.py
### 캡쳐할 것: 
1. 전체 코드 따라한 것
2. 추가로 softmax 계산 및 함수이용 비교
3. loss 값 (1.5608), 파일별로 확인한 결과, train_module.py 수행 결과
### 보고서 작성내용:
  - 대체과제 진행 총평

### Softmax 비교값 확인을 위한 코드
~~~ python
x=torch.FloatTensor([[1],[2],[3]])
F.softxmax(x,dim=0)
~~~
