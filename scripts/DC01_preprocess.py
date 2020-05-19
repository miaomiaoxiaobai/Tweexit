#!/usr/bin/python

"""
    Data preprocessing :
        - importation of an open source corpus of classified tweeets and related sentiment
        - removal of unused labels & NA vlaues
        - parsing and cleaning 
        - Save clean data
"""

# LIBRAIRIES
import re
import pandas        as     pd
from   bs4           import BeautifulSoup
from   nltk.tokenize import WordPunctTokenizer

# IMPORT DATA
data_open = pd.read_csv(
    'data_opendata.csv', 
    encoding = "ISO-8859-1", 
    header   = None,  
    names    = ['label', 'id', 'date', 'query', 'user', 'text']
)

# DROP UNUSED COLUMNS
data_open.drop(['id','date','query','user'], axis = 1, inplace = True)

# REPLACE LABELS
# 0: negative sentiment | +1: positive sentiment
data_open.loc[data_open['label'] == 0, 'label'] = 0
data_open.loc[data_open['label'] == 4, 'label'] = 1

# REMOVE NAS
data_open = data_open.dropna(axis = 0)

# CLEAN & PARSE TWEETS
tok          = WordPunctTokenizer()
pat1         = r'@[A-Za-z0-9]+'
pat2         = r'https?://[A-Za-z0-9./]+'
pat_combined = r'|'.join((pat1, pat2))

def tweet_cleaner(text):
    soup     = BeautifulSoup(text, 'lxml')
    souped   = soup.get_text()
    stripped = re.sub(pat_combined, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case   = letters_only.lower()
    words        = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

clean_tweet_texts = []

for _ in range(len(data_open)):
    clean_tweet_texts.append(tweet_cleaner(data_open['text'][_]))

# CLEAN DATAFRAME
data_clean = pd.DataFrame(clean_tweet_texts, columns = ['text'])
data_clean['label'] = data_open.label

# REMOVE NAS
data_clean = data_clean.dropna(axis = 0)

# EXPORT
data_clean.to_csv('data_clean.csv', encoding = 'utf-8')

# DEL
del data_clean, data_open, \
    tok, clean_tweet_texts, \
    pat1, pat2, pat_combined \