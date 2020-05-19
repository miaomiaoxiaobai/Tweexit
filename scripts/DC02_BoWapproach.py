#!/usr/bin/python

"""
    Bag of Word approach :
        - build sparse bow matrix
        - analyse word frequencies
        - keep relevant words (>threshold frequency)
        - export
"""

# LIBRAIRIES
import scipy
import pandas            as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud                       import WordCloud, STOPWORDS, ImageColorGenerator

# IMPORT CLEAN DATA
data_clean = pd.read_csv(
    'data_clean.csv', 
    encoding = "ISO-8859-1", 
    header   = 0,  
    names    = ['text', 'label']
)
data_clean = data_clean.dropna(axis = 0)

# SPLIT NEG / POS TWEETS
data_clean_neg = data_clean.loc[data_clean['label'] == 0]
data_clean_pos = data_clean.loc[data_clean['label'] == 1]

# GET WORD FREQUENCES
# function
def wordfrequences(corpus):
    corpus       = list(corpus)
    vec          = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    somme        = bag_of_words.sum(axis = 0)
    words_freq   = [(word, somme[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq   = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq
# word frequences

wfreq_    = wordfrequences(data_clean.iloc[:,0])

# neg / pos word frequences
wfreq_neg = wordfrequences(data_clean_neg.iloc[:,0])
wfreq_pos = wordfrequences(data_clean_pos.iloc[:,0])

# PLOT WORDCLOUDS
plots = False

# function
def cloudwordfreq(
    freqlist, limit, 
    colormap = "Blues",
    stop_w   = True,
    show     = False,
    save_as  = None):
    """
    freqlist = list of tuples(word, n_occurences)
    limit    = n_max of words to display
    """
    words   = [] ; weights = []
    for _ in range(limit*2):
        words.append(freqlist[_][0])
        weights.append(freqlist[_][1])
    if stop_w:
        stopwords = set(STOPWORDS)
        words = [word for word in words if word not in stopwords]
    words = words[:limit] ; words = " ".join(words)
    wordcloud = WordCloud(
        width = 1600, height = 800,
        background_color = "rgba(255, 255, 255, 0)", mode = "RGBA",
        relative_scaling = .5,
        colormap         = colormap
    ).generate(words)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    if save_as is not None: 
        plt.savefig(
            fname = '%s.png' % save_as, 
            dpi   = 300, 
            transparent = True
        )
    if show: plt.show()
    else:    plt.close()

# all / only positives / only negatives
if plots:
    cloudwordfreq(wfreq_,    limit = 25, save_as = "wordcloud")
    cloudwordfreq(wfreq_neg, limit = 25, save_as = "wordcloud_neg", colormap = "Reds")
    cloudwordfreq(wfreq_pos, limit = 25, save_as = "wordcloud_pos", colormap = "Greens")

# KEEP WORDS WITH AT LEAST 16 OCCURENCES
vectorizer = CountVectorizer(
    stop_words = 'english', 
    min_df     = 16
)

X_sparse = vectorizer.fit_transform(list(data_clean['text']))
X_dense  = X_sparse.todense()
y        = data_clean['label'].values

if len(X_dense) != len(y): raise SystemExit("...")

# EXPORT
scipy.sparse.save_npz('X_bofw.npz', X_sparse) ; np.savez('y_bofw.npz', y)

# DEL
del X_sparse, X_dense, y, \
    wfreq_, wfreq_pos, wfreq_neg, \
    data_clean, data_clean_neg, data_clean_pos \