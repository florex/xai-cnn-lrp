This project implements a new method to explain 1D-CNN especially for text classification task. Codes was designed for
experimental purposes and cannot be used in the state to handle any type of 1D-CNN without adaptation. The project comes with
a multi-channel 1D-CNN model generator which can be used to generating testing models.

Dependencies :
    - python 3.6+
    - keras (tested on 2.2+)
    - numpy (1.16+)
    - pandas (0.24+)

The project contains 4 main directories :

--data/sentiment_analysis
  This directory contains training and test data to build 1D-CNN models  and test the explanation method

-- models :
  This directory contains pretrained 1D-CNN models for sentiment analysis and for question answering task.

-- tokenizers :
  This directory contains saved keras tokenizer for different datasets. The tokenizer contains the vocabulary that was used
  to train the pretrained model associated to the dataset

-- explanations :
  This directory contains the results of explanations when executing the file explain_cnn.py. The explanations provided are contained in a json
  file associated to the name of the dataset/model to explain.

=======
Explaining a model
=======
- The file explain_cnn.py already has default value. python explain_cnn.py to execute the explainer with the default configurations.
- To explain another model on a set of samples, execute the following actions :
   1. open the file explain_cnn.py an edit the model parameters :
      - set the variable model_name to the model you want to explain. Pretrained model_names are : imdb, qa_1000, qa_5500, and merged
      - Set the variable n_classes to the number of classes.
        By default, the pretrained models was built with a word embedding dimension of 50 and a maximum number of words per sentence of 50
      - Set the variable class_names to the name of classes : class_names = ['NEGATIVE','POSITIVE'] for Sentiment analysis
        or class_names =  ['DESC','ENTY','ABBR','HUM','NUM','LOC'] for Question Answering
      - Set the variable embedding_dim to the dimension of the embedding vectors.
      - The variable kernel_sizes is a list of integers representing the  kernel_size per channel. Example : kernel_sizes = [1,2,3]
   2. run the command python explain_cnn.py
   3. The result of the explanation is contained in the directory explanations under the name : <model_name>_all_feature_ngrams.json

#To explain the model on a single sentence :
   edit and execute the file explain_sentence.py

The code implemented to explain 1D-CNN assumes that the CNN architecture taken as input has exactly 2 dense layers,
a variable number of channels (from 1 to n), a single global max-pooling layer, one convolution layer per channel
and a variable number of filters and kernel_sizes per channel.

NB: Further versions will take into account models with a variable number of dense layers.


=======
Explanation results
=======
The complete json file representing the explanation for a set of predictions is structured as follows :
The json file is in the form of a list of elements where each element represents the explanation of a particular input sentence
The json element representing the explanation of each sentence was designed to be self-explanatory. Contributions are in the form
- ngram feature :
   - CLASS_NAME : value

Overall represents the relevance of the feature to the class predicted as the difference between its contribution to this class
and the mean of its contribution to other classes except the predicted class.

[
    {
        "sentence": [
            "it was either too cold not enough flavor or just bad"
        ],
        "target_class": "NEGATIVE",
        "predicted_class": "NEGATIVE",
        "features": {
            "all": {
                "1-ngrams": [
                    {
                        "was": {
                            "NEGATIVE": 0.07160788029432297,
                            "POSITIVE": -0.06556335836648941,
                            "Overall": 0.13717123866081238
                        }
                    },
                ...
                ],
                "0-ngrams": [
                    {
                        "": {
                            "NEGATIVE": 0.1498018503189087,
                            "POSITIVE": -0.11607369035482407,
                            "Overall": 0.26587554812431335
                        }
                    }
                ]
            },
            "sufficient": {
                "1-ngrams": [
                    {
                        "not": 0.38679349422454834
                    }
                ]
            },
            "necessary": {}
        }
    },
    ...
   ]

"0-ngram" represents ngram features which are not in the vocabulary or the translation of 0-padding sequences

=======
Training a 1D-CNN model
=======
The project comes with codes to trained your own 1D-CNN.
To build your own CNN model
1. Defines the following parameters.

 - model_name : models names are defined in the variable file_path_dic. If you want to build a model from your
own dataset, make sure the data file is saved inside the directory data/sentiment analysis and that its format is as described in data/readme.txt
Also make sure to add an entry corresponding to the data file in the dictionary file_path_dic
 - embedding_dim : the dimension of word embeddings
 - n_classes : The number of classes in the dataset
 - max_words : The maximum number of words per sentence. Sentences with less word will be padded and sentences with higher number of words will
               be pruned to reach the max_words
 - kernel_sizes : a list of integers indicating the kernel_size per channel. Example : kernel_sizes = [1,2,3] means that they are 3 channels
   and the first channel has filters of kernel_size 1, the second channel has filters of kernel_sizes 2 and the third channel has filters of
   kernel sizes 3

 - n_filters : a list representing the number of filters per channel. Example : n_filters = [40,40,40]. It means that every channel has 40 filters
 which makes a total of 120 filters.
   NB : len(kernel_sizes) == len (n_filters)

2. After execute the command python train_1d_cnn.py
   The model will be saved in the directory models under a name related to the name defined in the variable model_name


