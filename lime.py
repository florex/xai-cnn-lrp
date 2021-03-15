from lime.lime.lime_text import LimeTextExplainer, TextDomainMapper
import numpy
from keras.models import Model
from keras.models import model_from_json
from preprocessor import Preprocessor
import json
def load_model(file_name) :
    json_file = open(file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

embedding_dim = 50
n_classes=2
max_words = 100
kernel_sizes=[1,2,3]
models_dir = './models/'
model_name = 'merged'
#class_names = ['DESC','ENTY','ABBR','HUM','NUM','LOC']
class_names = ['NEGATIVE','POSITIVE']
file_name = model_name+'_d'+str(embedding_dim)+'_l'+str(max_words)+'_'+str(len(kernel_sizes))+'ch'
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

def new_predict(sample):
    print(sample)
    seq = pre.get_pad_sequence(sample)
    print(seq)
    #sample = seq.reshape(1, len(sample))
    return model.predict([seq]*len(kernel_sizes))
id=3
explainer = LimeTextExplainer(class_names=class_names)

l = []
for x_t in X_test[:1000]:
    exp = explainer.explain_instance(text_instance=x_t,labels=range(n_classes),classifier_fn=new_predict, num_samples=50)
    seq = pre.get_pad_sequence(x_t)
    #print(seq)
    sent = pre.tokenizer.sequences_to_texts(seq)
    #print(sent)
    out = new_predict([x_t])
    #print(out)
    y = numpy.argmax(out)
    d = dict()
    for i in range(len(class_names)):
        #print(class_names[i])
        lt = exp.as_list(i)
        lt.sort(key=lambda x: x[1],reverse=True)
        d.update({class_names[i]:lt})
        #print(lt)
    l.append(d)
    #print(d)

refs = json.dumps(l, indent=4)
f = open("explanations/"+model_name+"_all_feature_ngrams_lime_1.json", "w")
f.write(refs)
f.close()