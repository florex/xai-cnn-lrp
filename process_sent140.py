import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
import csv
import re
filepath = 'data/sentiment_analysis/training.1600000.processed.noemoticon.csv'
df = pd.read_csv(filepath, names=['label', 'id', 'date', 'flag','user','sentence'], sep=',', encoding="utf8")
sentences = df['sentence'].values
labels = df['label'].values
writer = csv.writer(open("sent140.txt", 'w', newline='',encoding="utf8"), delimiter='\t')
max = 0
for sent,y in zip(sentences,labels) :
    label = 0
    sent = sent.replace('"',"")
    sent = sent.replace("\t"," ")
    sent = sent.replace("<br />"," ")
    sent = re.sub("^@\w+",'',sent).strip()
    if (len(sent)>max) :
        max = len(sent)
    #print (sent,y)
    if y == 4 :
        print("hit")
        label = 1
    elif y == 2 :
        label = 2
    if len(sent) < 50 :
        writer.writerow([sent,label])

print(max)