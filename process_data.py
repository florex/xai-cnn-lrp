from preprocessor import Preprocessor
pre = Preprocessor(source='sent140',max_words = 50,embedding_dim=50, n_classes=3)
pre.generate_embedding_dataset() # generate train and test set consisting of embedding matrices