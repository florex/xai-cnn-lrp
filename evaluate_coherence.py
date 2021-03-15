from explainer import TextCNNExplainer
from preprocessor import filepath_dict
from keras.models import model_from_json
from preprocessor import Preprocessor, filepath_dict
from random import sample
from keras.preprocessing.sequence import pad_sequences
import numpy
import json
import gensim

def load_model(file_name) :
    json_file = open(file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

def load_json_file(file_name) :
    json_file = open(file_name, 'r')
    data = json.load(json_file)
    json_file.close()
    return data

def to_phrase(sent,d) :
    f_set = d
    f_list = []
    if "1-ngrams" in f_set:
        grams = [list(x.keys())[0] for x in f_set["1-ngrams"]]
        f_list += grams

    if "2-ngrams" in f_set:
        grams = [list(x.keys())[0] for x in f_set["2-ngrams"]]
        f_list += grams

    if "3-ngrams" in f_set:
        grams = [list(x.keys())[0] for x in f_set["3-ngrams"]]
        f_list += grams

    f_list = " ".join(f_list).split(" ")
    s_list = list(set(f_list))
    l_sent = sent.split()
    s_list = [w for w in s_list if w.strip()!='' and w in l_sent and w in wv]
    s_list.sort(key = lambda x : l_sent.index(x))
    return s_list

models_dir = './models/'
#Define the parameters of the model to explain
embedding_dim = 50
n_classes = 6
max_words = 100 # maximum number of word per sentence
kernel_sizes = [1,2,3]
#test_file_path = filepath_dict['qa_5500']
model_name = 'qa_5500'

class_names =  ['DESC','ENTY','ABBR','HUM','NUM','LOC']
#class_names = ['NEGATIVE','POSITIVE'] # for sentiment analysis
file_name = model_name+'_d'+str(embedding_dim)+'_l'+str(max_words)+'_'+str(len(kernel_sizes))+'ch'
model = load_model(models_dir+file_name+".json")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# load weights into new model
model_file_path = models_dir+file_name
model.load_weights(models_dir+file_name+'.h5')
print("Model loaded from disk")

model.summary()
print("Loading w2vec model...")
wv = gensim.models.KeyedVectors.load_word2vec_format("glove.6B/glove.6B.50d.w2vformat.txt")
print("W2vec model loaded")
print("Initializing the preprocessor...")
pre = Preprocessor(source=model_name, max_words = max_words, embedding_dim=embedding_dim, n_classes=n_classes)
X_train, y_train, X_test, y_test = pre.get_sentences()
print("Preprocessor initialized")
file_name = "similar_data/"+model_name+"_sim_dict.json"
sim_dict = load_json_file(file_name)
#wv = gensim.models.KeyedVectors.load_word2vec_format("glove.6B/glove.6B.50d.w2vformat.txt")
explainer = TextCNNExplainer(pre.tokenizer, class_names)

n=0
coherence = []
for sent_id, closest_sents in sim_dict.items() :
    print(sent_id)
    sent = X_train[int(sent_id)].lower()
    seq = pre.tokenizer.texts_to_sequences([sent])
    seq = pad_sequences(seq, padding='post', maxlen=100)
    print("===========================================")
    s_0 = explainer.sufficient_feature_set(model,seq[0,:]) #get the sufficient features
    n_0 =  explainer.necessary_feature_set(model,seq[0,:]) #get necessary features
    sp_0 = to_phrase(sent,s_0)
    np_0 = to_phrase(sent,n_0)
    print("Analyzing coherence for sentence : ", sent, "sufficient set :", sp_0, "necessary set : ",np_0)
    if sp_0 == [] :
        continue

    #
    x = closest_sents[0]
    k = list(x.keys())[0]
    sent_1 = X_train[int(k)].lower()
    seq = pre.tokenizer.texts_to_sequences([sent_1])
    seq = pad_sequences(seq, padding='post', maxlen=100)
    s_1 = explainer.sufficient_feature_set(model, seq[0, :])  # get the sufficient features
    print(s_1)
    n_1 = explainer.necessary_feature_set(model, seq[0, :])  # get the necessary features
    sp_1 = to_phrase(sent_1, s_1)
    np_1 = to_phrase(sent_1, n_1)
    if sp_1 == [] :
        continue
    v_1 = wv.n_similarity([w for w in sent.split() if w in wv],[w for w in sent_1.split() if w in wv])
    s_1 = wv.n_similarity(sp_1, sp_0)
    if s_1 <= 0 :
        continue
    cpt = 2
    for x in closest_sents[1:] :
        k = list(x.keys())[0]
        sent_i = X_train[int(k)].lower()
        seq = pre.tokenizer.texts_to_sequences([sent_i])
        seq = pad_sequences(seq, padding='post', maxlen=100)
        s_i = explainer.sufficient_feature_set(model, seq[0,:])  #get the sufficient features
        n_i = explainer.necessary_feature_set(model, seq[0,:]) #get the necessary features
        sp_i = to_phrase(sent_i,s_i)
        np_i = to_phrase(sent_i,n_i)
        if sp_i == [] :
            continue
        v_i = wv.n_similarity([w for w in sent.split() if w in wv], [w for w in sent_i.split() if w in wv])
        s_i = wv.n_similarity(sp_i,sp_0)
        n += 1
        c = (float(s_i)/v_i)/(float(s_1)/v_1)
        coherence.append(c)
        arr = numpy.array(coherence)
        mean = numpy.mean(arr)
        std = numpy.std(arr)
        print("Closest sentence #" + str(cpt), sent_i)
        print("Sufficient set", sp_i)
        print("Necessary set", np_i)
        print("coherence", c)
        print("=========> Avg coherence : {0}, std : {1}".format(mean,std))
        cpt += 1
    print("**************************************")
    print(sent_id,closest_sents)