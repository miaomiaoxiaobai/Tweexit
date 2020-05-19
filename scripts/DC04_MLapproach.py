#!/usr/bin/python

"""
    Compare two classical machine learning models results to classify
    tweets as either positive sentiment or negative sentiment
        - Clean data importation
        - Split train/test
        - Train Naive Bayes & XGBoost
        - Compare results with confusion matrices
"""

# LIBRAIRIES
import scipy
import itertools
import datetime
import pandas            as pd
import matplotlib.pyplot as plt
import xgboost           as xgb
from scipy                   import sparse
from xgboost                 import XGBClassifier
from sklearn.metrics         import confusion_matrix, classification_report
from sklearn.naive_bayes     import MultinomialNB
from sklearn.model_selection import train_test_split

wordembedding = False
plots         = False
coefs         = False

# IMPORT DATA
if not wordembedding:
    X = scipy.sparse.load_npz('X_bofw.npz')
    y = np.load('y_bofw.npz') ; y = y.f.arr_0
else:
    X = np.load('X_wemb.npz') ; X = X.f.arr_0
    y = np.load('y_wemb.npz') ; y = y.f.arr_0

# CONFUSION MATRICES PLOT
def plot_cmatrix(
    cm, 
    classes,
    cmap      = plt.cm.Blues,
    normalize = True,
    save_as   = None,
    show      = True
):
    if normalize: cm = cm.astype('float')/cm.sum(axis = 1)[:,np.newaxis]
    ticks  = np.arange(len(classes))
    fmt    = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.colorbar()
    plt.xticks(ticks, classes, rotation = 45)
    plt.yticks(ticks, classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, 
            format(cm[i, j], fmt),
            horizontalalignment = 'center',
            color = 'red' if cm[i, j] > thresh else 'green'
        )
    plt.ylabel('Real label') ; plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_as is not None: 
        plt.savefig(
            fname = 'cm_{}_{}.png'\
                .format(save_as, 'normalized' if normalize else 'notnormalized'),
            dpi   = 300,
            transparent = True
        )   
    if not show: plt.close()

# SPLIT TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size    = 0.2,
    random_state = 10
)

# NAIVES BAYES
# Bench_1
t1 = datetime.datetime.now()
# Init
model = MultinomialNB()
# Fit
model = model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Bench_2
t2 = datetime.datetime.now() ; bench_NB = t2 - t1 ; del t1, t2
# Scores
if coefs:
    precision_NB = round(model.score(X_test, y_test)*100, 2)
    print("Benchmark : {}".format(bench_NB))
    print("Precision : {}%".format(precision_NB))
    print("Scores : ")
    print(classification_report(y_test, y_pred))
# Confusion matrix
if plots:
    plot_cmatrix(
        confusion_matrix(y_test, y_pred),
        classes = ['Positive', 'Negative'], 
        save_as = "NB"
    )

# XGBOOST
# Bench_1
t1 = datetime.datetime.now()
# Init
model = XGBClassifier(nthread = 8)
# Fit
model = model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Bench_2
t2 = datetime.datetime.now() ; bench_XGB = t2 - t1 ; del t1, t2
# Scores
if coefs:
    precision_XGB = round(model.score(X_test, y_test)*100, 2)
    print("Benchmark : {}".format(bench_XGB))
    print("Precision : {}%".format(precision_XGB))
    print("Scores : ")
    print(classification_report(y_test, y_pred))
# Confusion matrix
if plots:
    plot_cmatrix(
        confusion_matrix(y_test, y_pred),
        classes = ['Positive', 'Negative'],
        save_as = "XGB"
    )

# DEL
del model, \
    X, X_train, X_test, \
    y, y_train, y_test, y_pred \