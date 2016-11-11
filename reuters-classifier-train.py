from random import shuffle

import nltk
import numpy
from gensim.models import Doc2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from sklearn.preprocessing.label import MultiLabelBinarizer

nltk.download('reuters')
nltk.download('punkt')

google_news_word2vec_model_location = 'data/GoogleNews-vectors-negative300.bin.gz'
doc2vec_model_location = 'model/doc2vec-model.bin'
doc2vec_dimensions = 300
classifier_model_location = 'model/classifier-model.bin'

doc2vec = Doc2Vec.load(doc2vec_model_location)

# Convert the categories to one hot encoded categories
labelBinarizer = MultiLabelBinarizer()
labelBinarizer.fit([reuters.categories(fileId) for fileId in reuters.fileids()])

# Convert load the articles with their corresponding categories
train_articles = [{'raw': reuters.raw(fileId), 'categories': reuters.categories(fileId)} for fileId in reuters.fileids() if fileId.startswith('training/')]
test_articles = [{'raw': reuters.raw(fileId), 'categories': reuters.categories(fileId)} for fileId in reuters.fileids() if fileId.startswith('test/')]
shuffle(train_articles)
shuffle(test_articles)

# Convert the articles to document vectors using the doc2vec model
train_data = [doc2vec.infer_vector(word_tokenize(article['raw'])) for article in train_articles]
test_data = [doc2vec.infer_vector(word_tokenize(article['raw'])) for article in test_articles]
train_labels = labelBinarizer.transform([article['categories'] for article in train_articles])
test_labels = labelBinarizer.transform([article['categories'] for article in test_articles])
train_data, test_data, train_labels, test_labels = numpy.asarray(train_data), numpy.asarray(test_data), numpy.asarray(train_labels), numpy.asarray(test_labels)

# Initialize the neural network
model = Sequential()
model.add(Dense(input_dim=doc2vec_dimensions, output_dim=500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=1200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=400, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=600, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=train_labels.shape[1], activation='sigmoid'))
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Saves the model with highest score after each training cycle
checkpointer = ModelCheckpoint(filepath=classifier_model_location, verbose=1, save_best_only=True)

# Train the neural network
model.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=32, nb_epoch=15, callbacks=[checkpointer])