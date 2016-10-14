import nltk
from gensim.models import Doc2Vec
from keras.models import load_model
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize
from sklearn.preprocessing.label import MultiLabelBinarizer

nltk.download('reuters')
nltk.download('punkt')

doc2vec_model_location = 'model/doc2vec-model.bin'
classifier_model_location = 'model/classifier-model.bin'

doc2vec = Doc2Vec.load(doc2vec_model_location)

test_articles = [{'raw': reuters.raw(fileId), 'categories': reuters.categories(fileId)} for fileId in reuters.fileids() if fileId.startswith('test/')]

test_data = [doc2vec.infer_vector(word_tokenize(article['raw'])) for article in test_articles]
test_data = normalize(test_data)

# Initialize the neural network
model=load_model(classifier_model_location)

# Make predictions
predictions = model.predict(test_data)

# Enable and disable classes with a threshold
predictions[predictions<0.5] = 0
predictions[predictions>=0.5] = 1

# Convert enabled classes back to categorie names
labelBinarizer = MultiLabelBinarizer()
labelBinarizer.fit([reuters.categories(fileId) for fileId in reuters.fileids()])
predicted_labels = labelBinarizer.inverse_transform(predictions)

for predicted_label, test_article in zip(predicted_labels, test_articles):
    print('title: {}'.format(test_article['raw'].splitlines()[0]))
    print('predicted: {} - actual: {}'.format(list(predicted_label), test_article['categories']))
    print('')