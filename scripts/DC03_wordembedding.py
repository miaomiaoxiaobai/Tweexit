#!/usr/bin/python

"""
    Word embedding:
        - Stopwords & NAs removal
        - Building of Word2Vec model
        - Redefine dimension W2V space
"""

# LIBRAIRIES
import nltk
import gensim
import pandas        as pd
from   nltk.corpus   import stopwords
from   nltk.tokenize import WordPunctTokenizer

nltk.download('stopwords')

# IMPORT CLEAN DATA
data_clean = pd.read_csv(
    'data_clean.csv', 
    encoding = "ISO-8859-1",
    header   = 0,  
    names    = ['text', 'label']
)
# REMOVE NAS
data_clean = data_clean.dropna(axis = 0)

# IMPORT STOPWORDS
stopwords = stopwords.words('english')

# TOKENIZE & REMOVE STOPWORDS IN TWEETS
tok       = WordPunctTokenizer()
tweetstok = data_clean.iloc[:,0]
tweetstok = [tok.tokenize(x) for x in tweetstok]
tweetstok = [[word for word in tweet if word not in stopwords] for tweet in tweetstok]

# WORD2VEC MODEL
modelw2v = gensim.models.Word2Vec(
    tweetstok,
    size      = 150,
    window    = 10,
    min_count = 20,
    workers   = 8
)
modelw2v.train(tweetstok, total_examples = len(tweetstok), epochs = 10)

# EXPORT MODEL
modelw2v.save('modelw2v.bin')

# REDUCE TWEETS AS 150-DIMENSION-VECTORS USING WORD2VEC
def tweettovector(tweettokenized, dimensions):
    vector = np.zeros(dimensions).reshape((1, dimensions))
    count  = 0.
    for word in tweettokenized:
        try:
            vector += modelw2v[word].reshape((1, dimensions))
        except KeyError: 
            continue         # (-> case if word not in corpus)
    if count > 1:
        vector /= count
    return vector.tolist()

tweetsasvector = [[]] * len(tweetstok)

for _ in range(len(tweetstok)):
    tweetsasvector[_] = tweettovector(tweetstok[_], 150)[0]

tweetsasvector = pd.DataFrame(tweetsasvector)
tweetsasvector = tweetsasvector.values         # as matrix

# TARGET
y = data_clean['label'][0:].values

if len(tweetsasvector) != len(y): raise SystemExit ("...")

# EXPORT
np.savez('X_wemb.npz', tweetsasvector)
np.savez('y_wemb.npz', y)

# DEL 
del tweetsasvector, y, modelw2v, tok, stopwords, data_clean