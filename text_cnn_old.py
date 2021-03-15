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
import pandas as pd
numpy.set_printoptions(precision=2)
from numpy.linalg import norm


class TextCNN :
    def __init__(self, batch_size=32, text_length=50, embedding_dim=50, kernel_sizes = [1,2,3], n_filters = [100,100,100], n_class = 3):
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.text_length = text_length
        self.n_class = n_class
        self.model = None

    def train(self, X_train, y_train, X_test, y_test):
        pool = []
        inputs = []
        for kernel_size , n_filters in zip(self.kernel_sizes,self.n_filters) :
            input_shape=(self.text_length,self.embedding_dim)
            x = Input(shape=input_shape)
            C1D = layers.Conv1D(n_filters, kernel_size, activation='relu',input_shape=(self.text_length,self.embedding_dim))(x)
            MX1D = layers.GlobalMaxPool1D() (C1D)
            pool.append(MX1D)
            inputs.append(x)
        merged = pool[0]
        if len(self.kernel_sizes) > 1 :
            merged = concatenate(pool)
        dense1 = layers.Dense(10, activation='relu') (merged)
        dense2 = layers.Dense(10, activation='relu')(dense1)
        outputs = layers.Dense(self.n_class, activation='softmax') (dense2)
        self.model = Model(inputs=inputs, outputs=outputs)
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model.summary()
        history = self.model.fit([X_train]*len(self.kernel_sizes), y_train, epochs=100, verbose=True, validation_data=([X_test]*len(self.kernel_sizes),y_test), batch_size=self.batch_size)
        return self.model

    def evaluate(self, X_train, y_train, X_test, y_test):
        loss, accuracy = self.model.evaluate([X_train]*len(self.kernel_sizes), y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = self.model.evaluate([X_test]*len(self.kernel_sizes), y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    def save_model(self, model_name, model_dir='./models'):
        model_json = self.model.model.to_json()
        with open(model_dir+"/"+model_name+".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(model_dir+"/"+model_name+".h5")
            print("Saved model to disk")


model_name='sent140'
model_dir='models'
train_dataset = 'datasets/'+model_name+'/train.csv'
test_dataset = 'datasets/'+model_name+'/test.csv'
embedding_dim = 50
text_length = 50
n_class = 3
#train_data = numpy.loadtxt(train_dataset, delimiter=",", dtype=numpy.float32)
#test_data = numpy.loadtxt(test_dataset, delimiter=",", dtype=numpy.float32)
train_data = pd.read_csv(train_dataset, delimiter=",").values
test_data = pd.read_csv(test_dataset, delimiter=",").values
numpy.random.shuffle(train_data)
numpy.random.shuffle(test_data)
X_train = train_data[:,1:-n_class]
y_train = train_data[:,-n_class:]
X_test = test_data[:,1:-n_class]
y_test = test_data[:,-n_class:]

#normalize train an test matrices

train_norms = norm(X_train, axis=1, ord=2)
test_norms = norm(X_test, axis=1, ord=2)
train_norms[train_norms == 0] = 1
test_norms[test_norms == 0] = 1
print(train_norms)
X_train = X_train / train_norms.reshape(X_train.shape[0],1)
X_test = X_test / test_norms.reshape(X_test.shape[0],1)

print(X_train[:4,:10])
print(train_data[0,0],train_data[0,-1])
print(train_data[10,0],train_data[10,-2])
print(train_data[50,0],train_data[50,-1])
print(train_data[80,0],train_data[80,-2])
print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], text_length, embedding_dim))
X_test = X_test.reshape((X_test.shape[0], text_length, embedding_dim))

text_cnn = TextCNN(embedding_dim=embedding_dim,text_length=text_length)
model = text_cnn.train(X_train, y_train, X_test, y_test)
text_cnn.evaluate(X_train, y_train, X_test, y_test)
text_cnn.save_model(model_name)