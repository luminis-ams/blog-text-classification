import matplotlib.pyplot as plt
import numpy
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

google_news_word2vec_model_location = 'data/GoogleNews-vectors-negative300.bin.gz'

word2vec = Word2Vec().load_word2vec_format(google_news_word2vec_model_location, binary=True)
word2vec.init_sims(replace=True)

animals = '''alligator
ant
bat
bear
bee
bird
butterfly
baboon
camel
canine
cat
cheetah
chicken
chimpanzee
cow
cougar
crocodile
deer
dog
dolphin
donkey
duck
eagle
elephant
feline
fish
fox
frog
giraffe
goat
goldfish
gorilla
hamster
hippopotamus
horse
kangaroo
kitten
lion
lobster
mouse
monkey
octopus
orca
panda
pig
puppy
rabbit
rat
salmon
scorpion
seal
shark
shrimp
sheep
spider
squirrel
tiger
tuna
turtle
wasp
whale
wolf
zebra'''.split("\n")

vecs = numpy.asarray([word2vec[word] for word in animals])

tsneVecs = PCA(n_components=2).fit_transform(vecs)

fig, ax = plt.subplots()
ax.scatter(tsneVecs[:, 0], tsneVecs[:, 1])
for i, word in enumerate(animals):
    ax.annotate(word, (tsneVecs[:, 0][i], tsneVecs[:, 1][i]))
plt.show()