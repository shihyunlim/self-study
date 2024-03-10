# 초음파 광물 예측하기
## "학습셋과 테스트셋 구분" + "모델 저장과 재사용" 모두 적용한 코드!

from keras.models import Sequential, load_model
from keras.models import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv(' ', header=None)

dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=130, batch_size=5)
model.save('my_model.h5')

del model
model = load_model('my_model.h5')

print("\n Test Accuracy: %.4f" %(model.evaluate(X_test, Y_test)[1]))
```

## 초음파 광물 예측하기
> k겹 교차 검증하는 코드!

```python
from keras.models import Sequential
from keras.models import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv(' ', header=None)

dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy = []

for train, test in skf.split(X, Y):
	model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=100, batch_size=5) 
    k_accuracy = "%.4f" %(model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)
    
print("\n %.f fold accuracy:" %n_fold, accuracy)
