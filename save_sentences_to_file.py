import json
def load_json_file(file_name) :
    json_file = open(file_name, 'r')
    data = json.load(json_file)
    json_file.close()
    return data

models_dir = './models/'
#Define the parameters of the model to explain
kernel_sizes=[1,2,3]
model_name = 'merged'

count=1
file_name = "explanations/"+model_name+"_all_feature_ngrams_"+str(len(kernel_sizes))+"ch.json"
lrpa_ranking = load_json_file(file_name)

output_dir = './similar_data/'

for d in lrpa_ranking :
    sent = d["sentence"][0]
    fname = output_dir+str(count)+".txt"
    sufficient_set = d["features"]["sufficient"]
    sufficient_list = []
    if "1-ngrams" in sufficient_set :
        grams = [list(x.keys())[0] for x in sufficient_set["1-ngrams"]]
        sufficient_list += grams

    if "2-ngrams" in sufficient_set :
        grams = [list(x.keys())[0] for x in sufficient_set["2-ngrams"]]
        sufficient_list += grams

    if "3-ngrams" in sufficient_set :
        grams = [list(x.keys())[0] for x in sufficient_set["3-ngrams"]]
        sufficient_list += grams

    sufficient_list = " ".join(sufficient_list).split(" ")
    sufficient_phrase = " ".join([w for w in sent.split() if w in set(sufficient_list)])
    print(sent, set(sufficient_list))
    print(sufficient_phrase)

    necessary_set = d["features"]["necessary"]
    necessary_list = []
    if "1-ngrams" in necessary_set:
        grams = [list(x.keys())[0] for x in necessary_set["1-ngrams"]]
        necessary_list += grams

    if "2-ngrams" in necessary_set:
        grams = [list(x.keys())[0] for x in necessary_set["2-ngrams"]]
        necessary_list += grams

    if "3-ngrams" in necessary_set:
        grams = [list(x.keys())[0] for x in necessary_set["3-ngrams"]]
        necessary_list += grams

    necessary_list = " ".join(necessary_list).split(" ")
    necessary_phrase = " ".join([w for w in sent.split() if w in set(necessary_list)])
    print(sent, set(necessary_list))
    print(necessary_phrase)
    line = sent+":"+sufficient_phrase+":"+necessary_phrase
    with open(fname,"w", encoding="utf8") as f :
        f.write(str(count))
        f.write("\n")
        f.write(line)
    count+=1