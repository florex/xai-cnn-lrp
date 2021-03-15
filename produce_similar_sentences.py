from explainer import TextCNNExplainer
from preprocessor import filepath_dict
from keras.models import model_from_json
from preprocessor import Preprocessor, filepath_dict
from random import sample
import json
import gensim
import numpy

models_dir = './models/'
#Define the parameters of the model to explain
embedding_dim = 50
n_classes = 6
max_words = 100 # maximum number of word per sentence
kernel_sizes = [1,2,3]
#test_file_path = filepath_dict['qa_5500']
model_name = 'qa_5500'

def load_model(file_name) :
    json_file = open(file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

class_names =  ['DESC','ENTY','ABBR','HUM','NUM','LOC']
#class_names = ['NEGATIVE','POSITIVE'] # for sentiment analysis
file_name = model_name+'_d'+str(embedding_dim)+'_l'+str(max_words)+'_'+str(len(kernel_sizes))+'ch'
model = load_model(models_dir+file_name+".json")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# load weights into new model
model_file_path = models_dir+file_name
model.load_weights(models_dir+file_name+'.h5')
print("Model loaded from disk")

pre = Preprocessor(source=model_name, max_words = max_words, embedding_dim=embedding_dim, n_classes=n_classes)
X_train, y_train, X_test, y_test = pre.get_sequences()


output_prob = model.predict([X_train]*len(kernel_sizes))
y_predicted = numpy.argmax(output_prob, axis=1)

wv = gensim.models.KeyedVectors.load_word2vec_format("glove.6B/glove.6B.50d.w2vformat.txt")
wv.init_sims(replace=True)
sim_dict = {}
ord = 0
seen = set()
active = set(range(len(X_train)))
print(X_train.shape)
seed = sample(active,500)
cpt = 1
for i in seed :
    print("Proceed sample {0}".format(cpt))
    t_sent = pre.tokenizer.sequences_to_texts([X_train[i]])[0]
    t_y =  y_predicted[i]
    #print (t_y)
    d = []
    for j in active :
        if i != j :
            #sent = X_train[j]
            sent = pre.tokenizer.sequences_to_texts([X_train[j]])[0]
            y = y_predicted[i]
            s1 = [w for w in t_sent.split() if w in wv]
            s2 = [w for w in sent.split() if w in wv]
            if not y == t_y or s1 == [] or s2 == [] :
                continue
            s = wv.wmdistance(s1,s2)
            d.append((j,float(s)))
    d.sort(key=lambda x: x[1])
    d = [{k:v} for k,v in d[:10]]
    sim_dict.update({i:d})
    print("*********************************************")
    print(t_sent)
    print(d)
    cpt +=1
refs = json.dumps(sim_dict, indent=4)
f = open("similar_data/"+model_name+"_sim_dict.json", "w")
f.write(refs)
f.close()

