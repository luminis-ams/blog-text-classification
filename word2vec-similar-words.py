import matplotlib.pyplot as plt
import numpy
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

def makePlot(words):
    vecs = numpy.asarray([word2vec[word] for word in words])
    pca = PCA(n_components=2)
    pcaVecs = pca.fit_transform(vecs)
    fig, ax = plt.subplots()
    ax.scatter(pcaVecs[:,0], pcaVecs[:,1])
    for i, word in enumerate(words):
        ax.annotate(word, (pcaVecs[:,0][i], pcaVecs[:,1][i]))
    plt.show()

google_news_word2vec_model_location = 'data/GoogleNews-vectors-negative300.bin.gz'

word2vec = Word2Vec().load_word2vec_format(google_news_word2vec_model_location, binary=True)

makePlot(['cougar', 'tiger', 'cat', 'dog', 'feline', 'canine', 'mammal', 'fish', 'shark', 'tuna', 'monkey', 'primate', 'human', 'whale', 'bonobo', 'lioness', 'chimpanzee', 'gorilla', 'salmon', 'shrimp', 'lobster', 'baboon'])