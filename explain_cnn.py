from explainer_v1 import TextCNNExplainer
from preprocessor import filepath_dict
from keras.models import model_from_json
from preprocessor import Preprocessor, filepath_dict
import json
def load_model(file_name) :
    json_file = open(file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

models_dir = './models/'
#Define the parameters of the model to explain
embedding_dim = 50
n_classes = 2
max_words = 100 # maximum number of word per sentence
kernel_sizes = [1,2,3]
model_name = 'merged'
#test_file_path = filepath_dict['qa_5500']

#class_names =  ['DESC','ENTY','ABBR','HUM','NUM','LOC']
class_names = ['NEGATIVE','POSITIVE'] # for sentiment analysis
file_name = model_name+'_d'+str(embedding_dim)+'_l'+str(max_words)+'_'+str(len(kernel_sizes))+'ch'
model = load_model(models_dir+file_name+".json")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# load weights into new model
model_file_path = models_dir+file_name
model.load_weights(models_dir+file_name+'.h5')
print("Model loaded from disk")

model.summary()

pre = Preprocessor(source=model_name, max_words = max_words, embedding_dim=embedding_dim, n_classes=n_classes)
X_train, y_train, X_test, y_test = pre.get_sequences()
#X_test, y_test = pre.texts_to_sequences(test_file_path)

explainer = TextCNNExplainer(pre.tokenizer, class_names)
#Explain the first 10 instances of the test data.
explanations = explainer.compute_ngrams_contributions(model, X_test[0:1000], y_test[0:1000],rule='L2')
print(explainer.sufficient_feature_set(model, X_test[2]))
print(explainer.necessary_feature_set(model, X_test[2]))
refs = json.dumps(explanations, indent=4)
f = open("explanations/"+model_name+"_all_feature_ngrams_"+str(len(kernel_sizes))+"ch.json", "w")
f.write(refs)
f.close()

print("Explanations saved in the file: ./explanations/"+model_name+"_all_feature_ngrams_"+str(len(kernel_sizes))+"ch.json")
