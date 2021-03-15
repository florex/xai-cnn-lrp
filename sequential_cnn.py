from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras.layers.merge import concatenate
import numpy
from keras.models import Model
import tensorflow as tf
from keras.layers import Input
numpy.set_printoptions(precision=2)
import pandas as pd
from numpy.linalg import norm


model_name='yelp'
model_dir='models'
train_dataset = 'datasets/'+model_name+'/train.csv'
test_dataset = 'datasets/'+model_name+'/test.csv'
embedding_dim = 50
text_length = 50
n_class = 2

train_data = pd.read_csv(train_dataset, delimiter=",").values
test_data = pd.read_csv(test_dataset, delimiter=",").values

#train_data = numpy.loadtxt(train_dataset, delimiter=",", dtype=numpy.float32)
#test_data = numpy.loadtxt(test_dataset, delimiter=",", dtype=numpy.float32)
numpy.random.shuffle(train_data)
numpy.random.shuffle(test_data)
X_train = train_data[:,1:-n_class]
y_train = train_data[:,-n_class]
X_test = test_data[:,1:-n_class]
y_test = test_data[:,-n_class]

train_norms = norm(X_train, axis=1, ord=2)
test_norms = norm(X_test, axis=1, ord=2)
train_norms[train_norms == 0] = 1
test_norms[test_norms == 0] = 1
print(train_norms)
X_train = X_train / train_norms.reshape(X_train.shape[0],1)
X_test = X_test / test_norms.reshape(X_test.shape[0],1)

print(train_data[0,0],train_data[0,-2],train_data[0,-1])
print(train_data[10,0],train_data[10,-2],train_data[10,-1])
print(train_data[50,0],train_data[50,-2],train_data[50,-1])
print(train_data[80,0],train_data[80,-2],train_data[80,-1])
X_train = X_train.reshape((X_train.shape[0], text_length, embedding_dim))
X_test = X_test.reshape((X_test.shape[0], text_length, embedding_dim))

model = Sequential()
model.add(layers.Conv1D(100, 3, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(3, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=200, verbose=True,
                         validation_data=(X_test, y_test), batch_size=32)
model.summary()
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

