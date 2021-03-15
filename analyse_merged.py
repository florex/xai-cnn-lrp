import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
import csv
filepath = 'data/sentiment_analysis/merged_sent.txt'
df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t', encoding="utf8")
sentences = df.sentence.astype(str).values
labels = df['label'].values
total_not = 0
total_not_neg = 0
for sent,y in zip(sentences,labels) :
    sent = sent.lower()
    if "not" in sent.split() :
        total_not += 1
        if y==0 :
            total_not_neg += 1
        else :
            print(sent)

print(total_not, total_not_neg)