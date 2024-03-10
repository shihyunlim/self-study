# 아이리스 품종 예측하기

from tensorflow.keras.models import Sequential
from tenserflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# 시드 설정
np.random.seed(3)
tf.random.set_seed(3)

df = pd.read_csv(' ')

# 그래프로 확인
sns.pairplot(df, hue='species')
plt.show()

# 데이터 분류
dataset = df.values
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4]

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

# 모델 설정
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1)

print("\n Accuracy: %.4f" %(model.evaluate(X, Y_encoded)[1]))

