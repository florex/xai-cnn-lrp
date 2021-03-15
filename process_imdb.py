import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
import csv
filepath = 'data/sentiment_analysis/IMDB_Dataset.csv'
df = pd.read_csv(filepath, names=['sentence', 'label'], sep=',', encoding="utf8")
sentences = df['sentence'].values
labels = df['label'].values
writer = csv.writer(open("imdb_labelled_50000.csv", 'w', newline='',encoding="utf8"), delimiter='\t')
for sent,y in zip(sentences,labels) :
    label = 0
    sent = sent.replace('"',"")
    sent = sent.replace("\t"," ")
    sent = sent.replace("<br />"," ")
    if y == "positive" :
        label = 1
    writer.writerow([sent,label])

