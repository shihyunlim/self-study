# 경사 하강법을 이용해 최적의 a와 b값을 구하여 최적의 예측 직선을 구하는 방법

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부 시간 x와 성적 y의 리스트 만들기
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# 그래프로 나타내기
plt.figure(figsize=(8, 5))
plt.scatter(x,y)
plt.show()

# 리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸기(인덱스를 주어 하나씩 불러아 계산이 가능하게 하기 위함)
x_data = np.array(x)
y_data = np.array(y)

# 기울기 a와 절편 b 값 초기화
a = 0
b = 0

# 학습률 정하기
lr = 0.03

# 몇 번 반복될지 설정(0부터 세므로 원하는 반복 횟수+1)
epochs = 2001

# 경사 하강법 시작
for i in range(epochs): # 에포크 수만큼 반복
  y_pred = a * x_data + b # y를 구하는 식 세우기
  error = y_data - y_pred # 오차를 구하는 식
  # 오차 함수를 a로 미분한 값
  a_diff = -(2/len(x_data)) * sum(x_data * (error))
  # 오차 함수를 b로 미분한 값
  b_diff = -(2/len(x_data)) * sum(error)

  a = a - lr * a_diff # 학습률을 곱해 기존의 a값 업데이트
  b = b - lr * b_diff # 학습률을 곱해 기존의 b값 업데이트

  if i % 100 = 0: # 100번 반복될 때마다 현재 a, b 값 출력
    print("epoch=%.f, 기울기=%.04f, 절편=%.04f" %(i, a, b))

# 앞서 구한 기울기와 절편을 이용해 그래프를 다시 그리기
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_data), max(y_data)])
plt.show()
```

> 다중 선형 회귀 - 예측 직선 구하기
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits import mplot3d

# 공부 시간 x와 성적 y의 리스트 만들기
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

# 그래프로 확인
ax = plt.axes(projection='3d')
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')
ax.dist = 11
ax.scatter(x1, x2, y)
plt.show()

# 리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸기(인덱스를 주어 하나씩 불러아 계산이 가능하게 하기 위함)
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# 기울기 a와 절편 b 값 초기화
a1 = 0
a2 = 0
b = 0

# 학습률 정하기
lr = 0.02

# 몇 번 반복될지 결정
epochs = 2001

# 경사 하강법 시작
for i in range(epochs): # 에포크 수만큼 반복
  y_pred = a1 * x1_data + a2 * x2_data + b # y를 구하는 식 세우기
  error = y_data - y_pred # 오차를 구하는 식
  # 오차 함수를 a1으로 미분한 값
  a1_diff = -(2/len(x1_data)) * sum(x1_data * (error))
  # 오차 함수를 a2으로 미분한 값
  a2_diff = -(2/len(x2_data)) * sum(x2_data * (error))
  # 오차 함수를 b로 미분한 값
  b_diff = -(2/len(x1_data)) * sum(error)

  a1 = a1 - lr * a1_diff # 학습률을 곱해 기존의 a1값 업데이트
  a2 = a2 - lr * a2_diff # 학습률을 곱해 기존의 a2값 업데이트
  b = b - lr * b_diff # 학습률을 곱해 기존의 b값 업데이트

  if i % 100 = 0: # 100번 반복될 때마다 현재 a, b 값 출력
    print("epoch=%.f, 기울기1=%.04f, 기울기2=%.04f, 절편=%.04f" %(i, a1, a2, b))
