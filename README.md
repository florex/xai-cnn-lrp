# xai-cnn-lrp
This repository contains codes to explain One-Dimensional Convolutional Neural Networks (1D-CNN) using the principle explained in this article https://arxiv.org/abs/2010.03724. The explanation technique consists in computing the relevance of the various n-gram features and determining the sufficient and necessary n-grams. The project comes with a multi-channel 1D-CNN model generator which can be used to generate testing models. 

# Dependencies :
    - Anaconda (python 3.6)
    - keras (tested on 2.2.4)
    - tensorflow (1.13.1)
    - numpy (1.16)
    - pandas (0.24)

The project contains 4 main directories :

 - data/sentiment_analysis
   This directory contains training and test data to build 1D-CNN models and to test the explanation method

 -  models :
   This directory contains pretrained 1D-CNN models for sentiment analysis and for question answering tasks.

 -  tokenizers :
   This directory contains saved keras tokenizers for various datasets. A tokenizer contains the vocabulary that was used
   to build the pretrained model. 

 - explanations :
   This directory contains the results of explanations when executing the file explain_cnn.py. The explanations provided are saved in a json
   file which name is related to the name of the model to explain.

# Explaining a model

## Quick start :
The file explain_cnn.py already has default values. To execute the explainer with the default configurations. simply run the following command :
  
      python explain_cnn.py
      
By default, this command will compute the explanations related to the first 10 sentences of the "merged" dataset (yelp+amazon+imdb). The result will be in the file
./explanations/merged_all_features_ngrams.json

## Custom configurations :
For custom configurations, execute the following actions :
   1. open the file explain_cnn.py an edit the model parameters :
   
      - set the variable model_name to the model you want to explain. Pretrained model_names are : merged (sentiment analysis), qa_1000 and qa_5500 (Question answering)
      - Set the variable n_classes to the number of classes.
      - Set the embedding dimension (embedding_dim) and the maximum number of words per sentence.
        By default, the pretrained models was built with a word embedding dimension of 50 and a maximum number of words per sentence of 50
      - Set the variable class_names to the name of classes : 
        class_names = ['NEGATIVE','POSITIVE'] for Sentiment analysis
        or class_names =  ['DESC','ENTY','ABBR','HUM','NUM','LOC'] for Question Answering
      - Set the variable embedding_dim to the dimension of the embedding vectors.
      - The variable kernel_sizes is a list of integers representing the  kernel_size per channel. Example : kernel_sizes = [1,2,3]
      
      N.B : If you are not sure of which parameters to use for a particular model, then you should consider building a new model with your 
            own parameter (see the section : Training a 1D-CNN model below)
      
   2. run the command python explain_cnn.py
   
   3. The explanations will be generated in the directory explanations under the name : <model_name>_all_feature_ngrams.json

## To explain the model on a single sentence :
Edit the file explain_sentence.py and set the appropriate values for the parameters or leave defaults. Modify the variable "sent" with the sentence to explain
Then run the command :

    python explain_sentence.py

N.B : - The code implemented to explain 1D-CNN assumes that the CNN architecture consists of one or multiple input channel, one convolutional layer per channel, a single global max-pooling layer, a variable number of filters and kernel_sizes per channel and a variable number of hidden layer in the dense layer. 

# Model explanation results

The complete json file representing the explanation for a set of predictions is structured as follows :

        {
        "sentence": [
            "not frightening in the least and barely comprehensible"
        ],
        "target_class": "NEGATIVE",
        "predicted_class": "NEGATIVE",
        "features": {
            "all": {
                "1-ngrams": [
                    {
                        "barely": {
                            "NEGATIVE": -0.007197520695626736,
                            "POSITIVE": 0.0032649668864905834,
                            "Overall": -0.010462487116456032
                        }
                    },
                    {
                        "not": {
                            "NEGATIVE": 0.5278071761131287,
                            "POSITIVE": -0.4178711175918579,
                            "Overall": 0.9456782937049866
                        }
                    },
                   ....
                ],
                "0-ngrams": [
                    {
                        "": {
                            "NEGATIVE": 0.26937904953956604,
                            "POSITIVE": -0.2414221614599228,
                            "Overall": 0.5108011960983276
                        }
                    }
                ]
            },
            "sufficient": {
                "1-ngrams": [
                    {
                        "not": 0.9456782937049866
                    }
                ]
            },
            "necessary": {
                "1-ngrams": [
                    {
                        "not": 0.9456782937049866
                    }
                ]
            }
        }
    }

The json file is in the form of a list of elements where each element represents the explanation for a particular input sentence
which has been designed to be self-explanatory. Contributions are in the form
- ngram_feature :
   - CLASS_NAME : value

- The value of the key "Overall" represents the relevance of the feature to the class predicted as the difference between its contribution to the predicted class
and the mean of its contribution to other classes except the predicted class.

"0-ngram" represents ngram features which are not in the vocabulary or the translation of 0-padding sequences.


# Training a 1D-CNN model
The project comes with codes to train your own 1D-CNN.
To build your own CNN model for text classification do the followings :

1. Defines the model parameters.

     - model_name : models names are defined in the variable file_path_dic defined in the file preprocessor/Preprocessor.py. If you want to build a model from your
      own dataset, make sure the data file is saved inside the directory data/sentiment analysis and that its format is as described in data/readme.txt
      Also make sure to add an entry corresponding to the data file in the dictionary file_path_dic
     - embedding_dim : the dimension of word embeddings
     - n_classes : The number of classes in the dataset
     - max_words : The maximum number of words per sentence. Sentences with less word will be padded and sentences with higher number of words will
               be pruned to reach the max_words
     - kernel_sizes : a list of integers indicating the kernel_size per channel. Example : kernel_sizes = [1,2,3] means that they are 3 channels
       and the first channel has filters of kernel_size 1, the second channel has filters of kernel_sizes 2 and the third channel has filters of kernel sizes 3
     - n_filters : a list representing the number of filters per channel. Example : n_filters = [40,40,40]. It means that every channel has 40 filters
       which makes a total of 120 filters.
       NB : len(kernel_sizes) == len (n_filters)
     - hidden_layers : a list of integer representing the number of units per hidden layer in the fully connected stage. len(hidden_layers) is the number of layers in the fully connected neural network except the output layer. 

2. After execute the command :

       python train_1d_cnn.py
    
The model will be saved in the directory models under a name related to the name defined in the variable model_name

# To cite this :


    @misc{flambeau2020simplifying,
        title={Simplifying the explanation of deep neural networks with sufficient and necessary feature-sets: case of text classification}, 
        author={Jiechieu Kameni Florentin Flambeau and Tsopze Norbert},
        year={2020},
        eprint={2010.03724},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
