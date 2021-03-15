from lime.lime.lime_text import LimeTextExplainer, TextDomainMapper
from keras.preprocessing.sequence import pad_sequences
import numpy
from keras.models import Model
from keras.models import model_from_json
from preprocessor import Preprocessor
import time

def load_model(file_name) :
    json_file = open(file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

embedding_dim = 50
n_classes=6
max_words = 50

models_dir = './models/'
model_name = 'qa_5500'
class_names = ['DESC','ENTY','ABBR','HUM','NUM','LOC']
#class_names = ['POSTIVE','NEGATIVE']
file_name = model_name+'_d'+str(embedding_dim)+'_l'+str(max_words)
model = load_model(models_dir+file_name+".json")
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# load weights into new model
model_file_path = models_dir+file_name
model.load_weights(models_dir+file_name+'.h5')
print("Loaded model from disk")

model.summary()

pre = Preprocessor(source=model_name, max_words = max_words, embedding_dim=embedding_dim, n_classes=n_classes)
#X_train, y_train, X_test, y_test = pre.get_sequences()
X_train, y_train, X_test, y_test = pre.get_sentences()

"""
f = open("test_sent.txt","r")
sent = f.readlines()
print(sent)
seq = pre.tokenizer.texts_to_sequences([sent])
seq = pad_sequences(seq, padding='post', maxlen=50)
"""

def new_predict(sample):
    #print(sample)
    seq = pre.get_pad_sequence(sample)
    #print(seq)
    #sample = seq.reshape(1, len(sample))
    return model.predict(seq)
id=3
explainer = LimeTextExplainer(class_names=class_names)

start_time = time.time()
exp = explainer.explain_instance(text_instance=X_test[id],labels=[0,1,2,3,4,5],classifier_fn=new_predict, num_samples=100000)
print("--- %s seconds ---" % (time.time() - start_time))
"""
seq = pre.get_pad_sequence(X_test[id])
print(seq)
sent = pre.tokenizer.sequences_to_texts(seq)
print(sent)
out = new_predict([X_test[id]])
print(out)
y = numpy.argmax(out)
for i in range(6) :
    print(class_names[i])
    print(exp.as_list(i))
"""