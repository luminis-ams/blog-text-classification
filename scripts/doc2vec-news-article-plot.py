import itertools

import matplotlib.pyplot as plt
import nltk
from gensim.models import Doc2Vec
from nltk import word_tokenize
from nltk.corpus import reuters
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

nltk.download('reuters')
nltk.download('punkt')

google_news_word2vec_model_location = 'data/GoogleNews-vectors-negative300.bin.gz'
doc2vec_model_location = 'model/doc2vec-model.bin'
doc2vec_vectors_location = 'model/doc2vec-vectors.bin'
doc2vec_dimensions = 300

doc2vec = Doc2Vec.load(doc2vec_model_location)

jobs = [{'category': 'jobs', 'vec': doc2vec.infer_vector(word_tokenize(reuters.raw(fileId)))} for fileId in reuters.fileids(['jobs'])]
trade = [{'category': 'trade', 'vec': doc2vec.infer_vector(word_tokenize(reuters.raw(fileId)))} for fileId in reuters.fileids(['trade'])[:500]]

docs = [doc for doc in itertools.chain(jobs, trade)]

pca = PCA(n_components=50)
fiftyDimVecs = pca.fit_transform([doc['vec'] for doc in docs])
tsne = TSNE(n_components=2)
twoDimVecs = tsne.fit_transform(fiftyDimVecs)

fig, ax = plt.subplots()
for doc, twoDimVec in zip(docs, twoDimVecs):
    ax.scatter(twoDimVec[0], twoDimVec[1], color=('r' if doc['category'] == 'jobs' else 'b'))
plt.show()

