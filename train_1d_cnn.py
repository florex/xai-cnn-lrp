from preprocessor import Preprocessor
from text_cnn import TextCNN
#Define parameters :
model_name = 'merged' # other sentiment analysis predefined names include : imdb, yelp, amazon; QA models include : qa_1000 and qa_5500
embedding_dim = 50
n_classes=2  #change to 6 for question answering task
max_words = 100
kernel_sizes = [1,2,3]
n_filters = [40,40,40]

pre = Preprocessor(source=model_name,max_words = max_words,embedding_dim=embedding_dim, n_classes=n_classes)
X_train, y_train, X_test, y_test = pre.get_sequences(create_tokenizer=True)

#embedding_matrix = pre.create_embedding_matrix()

text_cnn = TextCNN(embedding_dim=embedding_dim,text_length=max_words, n_class=n_classes,kernel_sizes = kernel_sizes, n_filters = n_filters, batch_size=32,epochs=15,hidden_layers=[10,10,10])
model = text_cnn.train(X_train, y_train, X_test, y_test, None, len(pre.tokenizer.word_index) + 1)
model.summary()
text_cnn.evaluate(X_train, y_train, X_test, y_test)
text_cnn.save_model(model_name)