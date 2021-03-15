import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
import csv
import re
filepath = 'data/sentiment_analysis/QA/TREC_10.label.txt'
labels_ids_path = 'data/sentiment_analysis/QA/qa_label_ids.json'
def load_labels_ids(file_name) :
    if os.path.isfile(file_name):
        with open(file_name) as f:
            return json.load(f)
    else:
        return dict()
file = open(filepath, 'r', encoding="utf8")
lines = file.readlines()
last_id = 0
id = 0
labels_ids = load_labels_ids(labels_ids_path)
writer = csv.writer(open("data/sentiment_analysis/TREC_10.txt", 'w', newline='\n',encoding="utf8"), delimiter='\t')
for line in lines :
    line = line.strip()
    fields = line.split(" ")
    label = fields[0].split(":")[0]
    sent = ' '.join(fields[1:])
    if label in labels_ids :
        id = labels_ids[label]
    else :
        id = last_id
        last_id = last_id + 1
        print(id)
        labels_ids.update({label:id})
    writer.writerow([sent,id])

refs = json.dumps(labels_ids, indent=4)
f = open(labels_ids_path, "w")
f.write(refs)
f.close()
