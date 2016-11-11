from os import path
from random import shuffle

import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize

nltk.download('reuters')
nltk.download('punkt')

google_news_word2vec_model_location = 'data/GoogleNews-vectors-negative300.bin.gz'
doc2vec_model_location = 'model/doc2vec-model.bin'
doc2vec_vectors_location = 'model/doc2vec-vectors.bin'
doc2vec_dimensions = 300
classifier_model_location = 'model/classifier-model.bin'

# Load the reuters news articles and convert them to TaggedDocuments
taggedDocuments = [TaggedDocument(words=word_tokenize(reuters.raw(fileId)), tags=[i]) for i, fileId in enumerate(reuters.fileids())]
shuffle(taggedDocuments)

# Create and train the doc2vec model
doc2vec = Doc2Vec(size=doc2vec_dimensions, min_count=2, iter=10, workers=12)

# Build the word2vec model from the corpus
doc2vec.build_vocab(taggedDocuments)

# Load the google news word2vec model, should improve the models understanding of words (the Reuters 21578 dataset is not that big)
# Download from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
if(path.exists(google_news_word2vec_model_location)):
    doc2vec.intersect_word2vec_format(google_news_word2vec_model_location, binary=True)

doc2vec.train(taggedDocuments)
doc2vec.save(doc2vec_model_location)