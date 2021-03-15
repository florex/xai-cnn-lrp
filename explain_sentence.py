from preprocessor import Preprocessor, filepath_dict
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from explainer import TextCNNExplainer
def load_model(file_name) :
    json_file = open(file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

models_dir = './models/'

embedding_dim = 50
n_classes=2
max_words = 50
kernel_sizes=[1]
model_name = 'merged'
#test_file_path = filepath_dict['qa_5500']
#class_names =  ['DESC','ENTY','ABBR','HUM','NUM','LOC']
class_names = ['NEGATIVE','POSITIVE']
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

#sentence to explain
sent = "I don't like you"

seq = pre.tokenizer.texts_to_sequences([sent])
seq = pad_sequences(seq, padding='post', maxlen=50)
explainer = TextCNNExplainer(pre.tokenizer, class_names)

#explanations = explainer.compute_ngrams_contributions(model, X_test[0:10,:], y_test[0:10,:])
explanations = explainer.compute_ngrams_contributions(model, seq)
print(explanations)