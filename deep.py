import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPool1D, Dropout, Conv2D
import pandas as pd

X_train = np.load('xtrain.npy')
Y_train = np.load('ytrain.npy')
X_test = np.load('xtest.npy')
test_indices = np.load('test_indices.npy')

#X_train, X_test, Y_train, Y_test =  train_test_split(X, Y)

model = Sequential()
model.add(Dense(128, input_dim = X_train.shape[1], activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(96, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=150, batch_size=256)

prediction = model.predict(X_test)
print(prediction.shape)
predictions = np.array([1 if prediction[i] > 0.5 else 0 for i in range(prediction.shape[0])])
output = np.empty((len(predictions), 2))
output[:, 0] = test_indices.astype(int)
output[:, 1] = predictions.astype(int)
df = pd.DataFrame(output).astype(int)
df.to_csv('output_nn.csv',header=['id', 'spam'], index=False)


#score = model.evaluate(X_test, Y_test, batch_size=256)
#print(model.metrics_names)
#print(score)
