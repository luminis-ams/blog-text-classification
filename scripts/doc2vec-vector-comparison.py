import nltk
from gensim import matutils
from gensim.models import Doc2Vec
from nltk.corpus import reuters
from numpy import dot
from theano.gradient import np

nltk.download('reuters')
nltk.download('punkt')

doc2vec_model_location = 'model/doc2vec-model.bin'
doc2vec = Doc2Vec.load(doc2vec_model_location)

jobsVectors = [doc2vec.infer_vector(nltk.word_tokenize(reuters.raw(fileId))) for fileId in reuters.fileids('jobs')[:50]]
tradeVectors = [doc2vec.infer_vector(nltk.word_tokenize(reuters.raw(fileId))) for fileId in reuters.fileids('trade')[:50]]

def similarity(v1, v2):
    return dot(matutils.unitvec(v1), matutils.unitvec(v2))

previousJobsVector, previousTradeVector = None, None

jobsAndTradeSimilarities, jobsSimilarities, tradeSimilarities = [], [], []
for jobsVector, tradeVector in zip(jobsVectors, tradeVectors):

    jobsAndTradeSimilarities.append(similarity(jobsVector, tradeVector))
    if previousJobsVector is not None:
        jobsSimilarities.append(similarity(previousJobsVector, jobsVector))
    if previousTradeVector is not None:
        tradeSimilarities.append(similarity(previousTradeVector, tradeVector))
    previousJobsVector = jobsVector
    previousTradeVector = tradeVector

print(np.mean(jobsSimilarities))
print(np.mean(tradeSimilarities))
print(np.mean(jobsAndTradeSimilarities))