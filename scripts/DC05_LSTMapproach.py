#!/usr/bin/python

"""
    LSTM approach:
        - Import raw open data
        - Split data for training and testing
        - Create sequences for LSTM
        - Define LSTM architecture
        - Fit and evaluate model performances
"""

# LIBRAIRIES
import pandas                        as pd
import numpy                         as np
import matplotlib.pyplot             as plt
from   keras.preprocessing.text      import Tokenizer
from   keras.preprocessing.sequence  import pad_sequences
from   keras.models                  import Sequential
from   keras.layers                  import Dense, LSTM, Dropout, Activation
from   keras.layers.embeddings       import Embedding 
from   sklearn.model_selection       import train_test_split
from   keras.callbacks               import EarlyStopping 

# DATA
Opendata = pd.read_csv(
    'data_clean.csv', 
    encoding = "ISO-8859-1", 
    header   = 0,  
    names    = ['text', 'label']
)
Opendata = Opendata.dropna(axis = 0)

X = Opendata['text' ].values
y = Opendata['label'].values

y = np.asarray([min(1.0, p) for p in y])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2 ,
    random_state = 10
)

# CREATE SEQUENCES
vocabulary_size = 20000
tokenizer       = Tokenizer(num_words = vocabulary_size)
tokenizer.fit_on_texts(X_train)

sequences_X_train = tokenizer.texts_to_sequences(X_train)
data              = pad_sequences(sequences_X_train, maxlen = 30)
sequences_X_test  = tokenizer.texts_to_sequences(X_test)
data_test         = pad_sequences(sequences_X_test, maxlen  = 30)

def create_model(embedding_size, drop_rate = 0):
    model = Sequential()
    model.add(
        Embedding(
            vocabulary_size, 
            embedding_size, 
            input_length = 30
        )
    )
    if drop_rate > 0:
        model.add(Dropout(drop_rate))
    # LSTM
    model.add(LSTM(embedding_size))
    # OUTPUT
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(
        loss      = 'binary_crossentropy', 
        optimizer = 'adam', 
        metrics   = ['accuracy']
    )
    return model

early_stopping_monitor = EarlyStopping(patience = 3) 

# MODEL FIT
modele = create_model(100, 0.5)
history = modele.fit(
    data, y_train, 
    validation_split = 0.2, 
    epochs           = 10,
    callbacks        = [early_stopping_monitor]
)

plots  = False
scores = False

# PLOTS
if plots:
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()

# SCORES
if scores:
    scores = modele.evaluate(data_test, y_test, verbose = 0)
    print("Accuracy modele: %.2f%%" % (scores[1]*100))
    modele.summary()
