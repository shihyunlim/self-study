# 모딥 3장 선형 회귀
## 최소 제곱법으로 예측 직선 구하기

import numpy as np

# x값과 y값
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# x와 y의 평균값
mx = np.mean(x)
my = np.mean(y)

# 기울기 공식의 분모
divisor = sum([(mx - i)**2 for i in x])

# 기울기 공식의 분자
def top(x, mx, y, my):
  d = 0
  for i in range(len(x)):
    d += (x[i] - mx) * (y[i] - my)
  return d
dividend = top(x, mx, y, my)

# 기울기와 y 절편 구하기
a = dividend / divisor
b = my - (mx * a)

# 출력은 알아서!
```

> 평균 제곱 오차
```python
import numpy as np

# 기울기 a와 y 절편 b
fake_a_b = [3, 76]

# x와 y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data] # 파이썬에서 i[0]는 값 중 첫 번째를, i[1]은 두 번째를 의미
y = [i[1] for i in data]

# y=ax+b에 a와 b값을 대입하여 결과를 출력하는 함수
def predict(x):
  return fake_a_b[0]**x + fake_a_b[1]

# MSE 함수
def mse(y, y_hat):
  return ((y - y_hat) ** 2).mean())

# mse 함수를 각 y값에 대입하여 최종 값을 구하는 함수
def mse_val(y, predict_result):
  return mse(np.array(y), np.array(predict_result))

# 예측값이 들어갈 빈 리스트
predict_result = []

# 모든 x 값을 한 번씩 대입하여
for i in range(len(x)):
  # predict_result 리스트를 완성
  predict_result.append(predict(x[i]))

# 최종 MSE 출력
print("mse 최종값: " + str(mse_val(y, predict_result))
