from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from stop_words import get_stop_words
en_stop = get_stop_words('en')

from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()

from gensim import corpora, models
import gensim

#read data as string
with open('data/Bible.txt', 'r') as file1:
	bible_raw = file1.read().decode('utf8')

with open('data/Pride and Prejudice.txt', 'r') as file2:
	pride_raw = file2.read().decode('utf8')

with open('data/Autobiography of Benjamin Franklin.txt', 'r') as file3:
	franklin_raw = file3.read().decode('utf8')

#clean data
def clean(text):
	tokens = tokenizer.tokenize(text.lower())
	stopped = [i for i in tokens if i not in en_stop]
	no_number = [i for i in stopped if not i.isdigit()]
	#stemmed = [p_stemmer.stem(i) for i in stopped]
	return no_number

texts = [clean(bible_raw), clean(pride_raw), clean(franklin_raw)]

#assign id to each token while counting token for each doc
dictionary = corpora.Dictionary(texts)
#convert dictionary into bag-of-words, result a list of vectors [(term id, term freq)] 
corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=40)

print(ldamodel.print_topics(num_topics=3, num_words=10))


 


