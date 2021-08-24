from explainer_v1 import TextCNNExplainer
from preprocessor import filepath_dict
from keras.models import model_from_json
from preprocessor import Preprocessor, filepath_dict
import json
from keras.models import Model
import numpy

def load_model(file_name) :
    json_file = open(file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

models_dir = './models/'
#Define the parameters of the model to explain
embedding_dim = 50
max_words = 100 # maximum number of word per sentence
kernel_sizes=[1,2,3]
model_name = 'qa_5500'
if model_name == 'qa_5500' :
    n_classes = 6
    class_names = ['DESC', 'ENTY', 'ABBR', 'HUM', 'NUM', 'LOC']
elif model_name == 'merged' :
    n_classes = 2
    class_names = ['NEGATIVE', 'POSITIVE']  # for sentiment analysis

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
def load_json_file(file_name) :
    json_file = open(file_name, 'r')
    data = json.load(json_file)
    json_file.close()
    return data

data = X_test
explainer = TextCNNExplainer(pre.tokenizer, class_names)
#Explain the first 10 instances of the test data.
contributions = explainer.compute_contributions(model, X_test)
conv_layer_name = 'global_max_pooling1d_1'
conv_layers = conv_layers = ['conv1d_'+str(i) for i in range(1,len(kernel_sizes)+1)]
max_pool = Model(inputs=model.input, outputs=model.get_layer(conv_layer_name).output)
max_out = max_pool.predict([data]*len(kernel_sizes))
output_prob = model.predict([data]*len(kernel_sizes))
count = 0
"""
for d,c, prob in zip(max_out,contributions,output_prob) :
    d = d.reshape((d.shape[0], 1))
    out = d*c;
    approx = numpy.sum(out, axis=0)
    pred_class = numpy.argmax(prob)
    approx_class = numpy.argmax(approx)
    if(pred_class==approx_class) :
        count+=1
    #print("predicted outputs = "+str(pred_class)+ "approximated = "+str(approx_class))

fidelity = count*1.0/data.shape[0]

print("fidelity", fidelity)
print("max-pooling", max_out.shape)
print("Contributions ",contributions.shape)
"""
print("compute lime fidelity")
count = 0
file_name = "explanations/"+model_name+"_all_feature_ngrams_lime_1.json"
lime_ranking = load_json_file(file_name)
for x,prob in zip(lime_ranking,output_prob) :
    out = [0]*len(class_names)
    for y in range(len(class_names)) :
        for w,val in x[class_names[y]] :
            out[y] += val

    pred_class = numpy.argmax(prob)
    approx_class = numpy.argmax(out)
    if (pred_class == approx_class):
        count += 1

    #print("predicted outputs = " + str(pred_class) + "approximated = " + str(approx_class))
fidelity = count*1.0/len(lime_ranking)
print("fidelity", fidelity)
print("max-pooling", max_out.shape)
print("Contributions ",contributions.shape)

count=0
file_name = "explanations/"+model_name+"_all_feature_ngrams_"+str(len(kernel_sizes))+"ch.json"
lrpa_ranking = load_json_file(file_name)
print("Compute LRPA fidelity")
for x, prob in zip(lrpa_ranking,output_prob) :
    out = [0] * len(class_names)
    for y in range(len(class_names)):
        r1 = x["features"]["all"]["1-ngrams"]
        if "2-ngrams" in x["features"]["all"]:
            r1 = r1+ x["features"]["all"]["2-ngrams"]
        if "3-ngrams" in x["features"]["all"]:
            r1 = r1+ x["features"]["all"]["3-ngrams"]
        if "0-ngrams" in x["features"]["all"]:
            r1 = r1 + x["features"]["all"]["0-ngrams"]
        for d in r1 :
            v = list(d.items())[0][1]
            out[y] += v[class_names[y]]

    pred_class = numpy.argmax(prob)
    approx_class = numpy.argmax(out)
    if (pred_class == approx_class):
        count += 1

    #print("predicted outputs = " + str(pred_class) + "approximated = " + str(approx_class))
fidelity = count*1.0/len(lrpa_ranking)
print("fidelity", fidelity)
print("max-pooling", max_out.shape)
print("Contributions ",contributions.shape)
