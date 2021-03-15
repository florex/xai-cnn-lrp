import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
import csv
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
filepath_dict = {'yelp':   'data/sentiment_analysis/yelp_labelled.txt',
                 'amazon': 'data/sentiment_analysis/amazon_cells_labelled.txt',
                 'imdb':   'data/sentiment_analysis/imdb_labelled.txt',
                 'merged' : 'data/sentiment_analysis/merged_sent.txt',
                    'qa_1000': 'data/sentiment_analysis/qa_1000.txt',
'TREC_10': 'data/sentiment_analysis/TREC_10.txt',
'qa_5500': 'data/sentiment_analysis/qa_5500.txt',
                 }


# This class transforms sentences into embedding matrices for text classification
# The input file consists of a list of sentences follow by the class of the sentences
# set gen to True to force the generation of the embedding model
tokenizer_path = "./tokenizers/"
class Preprocessor :
    def __init__(self,source='yelp', embedding_dim=50, max_words = 100, gen = True, n_classes=2, embedding_dir = 'data/glove_word_embeddings', tokenizer_file=None) :
        print("Initializing the preprocessor....")
        self.embedding_dim = embedding_dim # number of column of the embedding matrix
        self.max_words = max_words # number of lines of the embedding matrix
        self.embedding_model_file = embedding_dir+'/glove.6B.'+str(self.embedding_dim)+'d.txt'
        self.n_classes = n_classes
        self.gen = gen
        self.source = source
        self.df = self.load_data_sources(source)
        #self.load_models()
        self.tokenizer = self.load_tokenizer(tokenizer_file)
        self.datasets_path = './datasets/'+self.source
        print("Preprocessor initialized")

    def load_tokenizer(self, file_name = None):
        if file_name is None :
            file_name = tokenizer_path + "/" + self.source + '_tokenizer.json'

        if os.path.isfile(file_name) :
            with open(file_name) as f:
                data = json.load(f)
                tokenizer = tokenizer_from_json(data)
                return tokenizer
        else :
            return Tokenizer(num_words=10000)

    def save_tokenizer(self,file_name=''):
        tokenizer_json = self.tokenizer.to_json()
        with open(tokenizer_path + "/" + self.source + '_tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    # this method load the basic utils models from files if exists otherwise they are generated.
    # set the property gen to True to force the generation of those models
    def load_models(self,source = None):
        if self.gen or os.path.isfile('./embedding_dic.json'):
            # update the embedding dic
            self.embedding_dic = self.create_filtered_embedding_dic()
        else :
            self.embedding_dic = self.loadDicModel('./embedding_dic.json')


    def load_all_data_sources(self):
        df_list = []
        for source, filepath in filepath_dict.items():
            df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t', encoding="utf8")
            df['source'] = source  # Add another column filled with the source name
            df_list.append(df)
        self.df = pd.concat(df_list)
        return self.df

    def load_data_sources(self,source) :
        self.df = self.load_all_data_sources()
        self.df = self.df[self.df['source'] == source]
        return self.df


    def loadDicModel(self, file):
        with open(file) as json_file:
            return json.load(json_file)

    def sentence_to_matrix(self, sentence):
        embedding_matrix = np.zeros((self.max_words,self.embedding_dim))
        i = 0
        for word in sentence :
            if word in self.embedding_dic :
                embedding_matrix[i] = np.array(self.embedding_dic[word])
                i+=1
            if i == self.max_words :
                break
        return embedding_matrix

    def generate_embedding_dataset(self):
        print("Start dataset generation...")
        sentences = self.df['sentence'].values
        y = dy = self.df['label'].values
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25,
                                                                            random_state=1000)
        # generate the embedding dataset for train sentences
        print("Generating train set...")
        self.generate_dataset_matrix(sentences_train,y_train,self.datasets_path+"/train.csv",self.datasets_path+"/train_sents_ref.json")

        #generate embedding dataset for test sentences
        print("Generating test set...")
        self.generate_dataset_matrix(sentences_test, y_test ,self.datasets_path+"/test.csv",self.datasets_path+"/test_sents_ref.json", test=True)
        print("Datasets generated")
        print("Train set : "+self.datasets_path+"/train.csv")
        print("Test set : "+self.datasets_path+"/test.csv")

    def generate_dataset_matrix(self, sentences, y_vect, output_file, sent_ref_file, test=False):
        writer = csv.writer(open(output_file, 'w', newline=''), delimiter=',')
        matrix = np.empty((0, self.embedding_dim * self.max_words + self.n_classes + 1), dtype=np.float16)
        i = 0  # index of the sentence.
        sentences_ref = dict()
        hope = len(sentences)/100
        for sent,y in zip(sentences,y_vect) :
            matrix = self.sentence_to_matrix(sent)
            out = [0]*self.n_classes
            #print(y, sent)
            out[y] = 1
            sentences_ref.update({i:{'sentence':sent,'class':int(y)}})
            matrix = np.insert(np.append(matrix.reshape((1, self.embedding_dim * self.max_words)), out), 0, i)
            matrix = matrix.reshape(1, 1 + self.n_classes + self.embedding_dim * self.max_words)
            writer.writerow(['{:.3f}'.format(x) for x in matrix.flatten()])
            #compute the progession
            i+=1
            percentage = i/hope
            if i%hope == 0 :
                print(i,out,sent)
                print(str(percentage)+"% completed")

            if (not test and i==200000) or (test and i==50000) :
                break;

        refs = json.dumps(sentences_ref, indent=4)
        f = open(sent_ref_file, "w")
        f.write(refs)
        f.close()

    def create_embedding_dic(self):
        embed_dic = dict()
        with open(self.embedding_model_file, encoding="utf8") as f:
            for line in f :
                word, *vector = line.split()
                wv = np.array(vector, dtype=np.float16)[:self.embedding_dim]
                embed_dic.update({word: wv.tolist()})
        refs = json.dumps(embed_dic, indent=4)
        f = open("embedding_dic.json", "w")
        f.write(refs)
        f.close()

    def get_sequences(self, create_tokenizer=False) :
        sentences = self.df.sentence.astype(str).values
        y = dy = self.df['label'].values
        out = np.zeros((len(y), self.n_classes))
        for i in range(len(y)) :
            out[i,y[i]] = 1
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, out, test_size=0.25,random_state=1000)
        if create_tokenizer :
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(sentences)
            self.save_tokenizer()
        X_train = self.tokenizer.texts_to_sequences(sentences_train)
        X_test = self.tokenizer.texts_to_sequences(sentences_test)
        self.vocab_size = len( self.tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        maxlen = self.max_words
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
        return(X_train, y_train, X_test, y_test)

    def texts_to_sequences(self,file_path) :
        df = pd.read_csv(file_path, names=['sentence', 'label'], sep='\t', encoding="utf8")
        sentences = df.sentence.astype(str).values
        y = dy = df['label'].values
        out = np.zeros((len(y), self.n_classes))
        for i in range(len(y)) :
            out[i,y[i]] = 1
        X_test = self.tokenizer.texts_to_sequences(sentences)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.max_words)
        return(X_test, out)

    def get_pad_sequence(self,text):
        seq = self.tokenizer.texts_to_sequences(text)
        pad_seq = pad_sequences(seq, padding='post', maxlen=self.max_words)
        return pad_seq


    def get_sentences(self, create_tokenizer=False) :
        sentences = self.df.sentence.astype(str).values
        y = dy = self.df['label'].values
        out = np.zeros((len(y), self.n_classes))
        for i in range(len(y)) :
            out[i,y[i]] = 1
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, out, test_size=0.25,random_state=1000)
        return(sentences_train, y_train, sentences_test, y_test)

    def create_filtered_embedding_dic(self):
        embed_dic = dict()
        vocab = self.vocabulary()
        with open(self.embedding_model_file, encoding="utf8") as f:
            for line in f :
                word, *vector = line.split()
                if word in vocab :
                    wv = np.array(vector, dtype=np.float16)[:self.embedding_dim]
                    embed_dic.update({word:wv.tolist()})
        refs = json.dumps(embed_dic, indent=4)
        f = open("embedding_dic.json", "w")
        f.write(refs)
        f.close()
        return embed_dic

    def create_embedding_matrix(self):
        vocab_size = len(self.tokenizer.word_index) + 1  # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        with open(self.embedding_model_file, encoding="utf8") as f:
            for line in f:
                word, *vector = line.split()
                if word in self.tokenizer.word_index:
                    idx = self.tokenizer.word_index[word]
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:self.embedding_dim]

        return embedding_matrix

    def vocabulary(self):
        tokenizer = Tokenizer()
        sentences = self.df['sentence'].values
        tokenizer.fit_on_texts(sentences)
        return tokenizer.word_index

    def filter_embedding_dic(self) :
        pass

