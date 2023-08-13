import time
import numpy as np
from numpy.random import seed
import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def pred(x_test):
    y_pred = model.predict(x_test)                          # Trả về kq dạng thường
    # print(y_pred)
    y_pred = to_categorical(np.argmax(y_pred, axis=1), 3)   # Đưa vè về kq dạng One Hot
    y_pred = np.argmax(y_pred, axis=1)                      # Đưa về về kq dạng numeric label
    return y_pred

def get_metrics(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro'), \
           recall_score(y_true, y_pred, average='macro'), \
           f1_score(y_true, y_pred, average='macro')

# Đặt seed
seed(1)
tensorflow.random.set_seed(2)

num_nodes = 6
optimizer = Adam(learning_rate=0.001)
loss_func = 'categorical_crossentropy'
metrics = ['accuracy']

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125)

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
# y_val_encoded = to_categorical(y_val)

# ======================================================================================
# Network Model
num_layers = 3

# Create the network
model = Sequential()

model.add(keras.Input(shape=(4,)))
for i in range(1, num_layers+1):
  model.add(Dense(num_nodes, activation='relu'))
  model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

# print('Neural Network Model Summary: ')
# print(model.summary())

# Compile the network
model.compile(optimizer,
              loss=loss_func,
              metrics=metrics)

start = time.time()
model.fit(X_train, y_train_encoded, batch_size=5, epochs=200, verbose=1)
stop = time.time()
train_time = stop - start

loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)

y_pred = pred(X_test)
metrics = get_metrics(y_test, y_pred)

# Print the test result
print('Loss: \t\t', loss,
      '\nAccuracy: \t', accuracy,
      '\nPrecision: \t', metrics[0],
      '\nRecall: \t', metrics[1],
      '\nF1_score: \t', metrics[2]
      )

# print('-'*30)
# print(y_pred)
# print(y_test)
# print(train_time)
