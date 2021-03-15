import json

def load_json_file(file_name) :
    json_file = open(file_name, 'r')
    data = json.load(json_file)
    json_file.close()
    return data

lrpa_ranking = load_json_file("explanations/qa_5500_all_feature_ngrams.json")
lime_ranking = load_json_file("explanations/qa_5500_all_feature_ngrams_lime.json")

c="LOC"
for x,y in zip(lrpa_ranking,lime_ranking) :
    r1 = x["features"]["all"]["1-ngrams"]
    r1 = [(k,v[c]) for d in r1 for k,v in d.items()]
    r2 = [(d[0],d[1]) for d in y[c] ]

    r1.sort(key=lambda x: x[1], reverse=True)
    r2.sort(key=lambda x: x[1], reverse=True)
    print ("LRP", r1)
    print ("LIME", r2)
    print("****************************************")



print (lime_ranking)