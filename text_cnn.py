from keras import layers
from keras import optimizers
from keras.layers.merge import concatenate
import numpy
from keras.models import Model
from keras.layers import Input
numpy.set_printoptions(precision=2)

class TextCNN :
    def __init__(self, batch_size=32, epochs = 50, text_length=100, embedding_dim=100, kernel_sizes = [1], n_filters = [40], hidden_layers = [10,10], n_class = 2):
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.text_length = text_length
        self.n_class = n_class
        self.model = None
        self.epochs = epochs
        self.hidden_layers = hidden_layers

    def train(self, X_train, y_train, X_test, y_test, embedding_matrix, vocab_size):
        pool = []
        inputs = []
        for kernel_size , n_filters in zip(self.kernel_sizes,self.n_filters) :
            input_shape=(self.text_length,self.embedding_dim)
            x = Input(shape=(self.text_length,))
            embedding = layers.Embedding(vocab_size, self.embedding_dim, input_length=self.text_length)(x)
            C1D = layers.Conv1D(n_filters, kernel_size, activation='relu',input_shape=(self.text_length,self.embedding_dim))(embedding)
            MX1D = layers.GlobalMaxPool1D() (C1D)
            pool.append(MX1D)
            inputs.append(x)
        merged = pool[0]
        if len(self.kernel_sizes) > 1 :
            merged = concatenate(pool)

        if len(self.hidden_layers) > 0 :
            dense = layers.Dense(self.hidden_layers[0], activation='relu') (merged)
            for n_unit in self.hidden_layers[1:] :
                dense = layers.Dense(n_unit, activation='relu') (dense)
            #dense2 = layers.Dense(10, activation='relu')(dense1)
            outputs = layers.Dense(self.n_class, activation='softmax') (dense)
        else :
            outputs = layers.Dense(self.n_class, activation='softmax')(merged)
        self.model = Model(inputs=inputs, outputs=outputs)
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model.summary()
        history = self.model.fit([X_train]*len(self.kernel_sizes), y_train, epochs=self.epochs, verbose=True, validation_data=([X_test]*len(self.kernel_sizes),y_test), batch_size=self.batch_size)
        return self.model

    def evaluate(self, X_train, y_train, X_test, y_test):
        loss, accuracy = self.model.evaluate([X_train]*len(self.kernel_sizes), y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = self.model.evaluate([X_test]*len(self.kernel_sizes), y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    def save_model(self, model_name, model_dir='./models'):
        model_json = self.model.to_json()

        with open(model_dir+"/"+model_name+"_d"+str(self.embedding_dim)+"_l"+str(self.text_length)+'_'+str(len(self.kernel_sizes))+"ch.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(model_dir+"/"+model_name+"_d"+str(self.embedding_dim)+"_l"+str(self.text_length)+'_'+str(len(self.kernel_sizes))+"ch.h5")
            print("Saved model to disk")


