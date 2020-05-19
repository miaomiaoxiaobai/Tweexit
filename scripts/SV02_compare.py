#!/usr/bin/python

"""
    Compare the evolutions of the sentiments expressed on filtered 
    streamed twitts time serie and the LSE time series
"""

# LIBRAIRIES
import pandas                 as pd
import datetime               as dt
import matplotlib             as mpl
import matplotlib.pyplot      as plt
from   sklearn.preprocessing  import PolynomialFeatures
from   sklearn.linear_model   import LinearRegression
from   matplotlib.dates       import DateFormatter
from   sklearn                import preprocessing
from   cassandra.cluster      import Cluster

# IMPORT LONDONSTOCKEXCHANGE
session = Cluster().connect('twitter') ; session.execute('USE twitter')
query   = 'SELECT * FROM LSE'
lse     = session.execute(query, timeout = None)

# TIME SPAN
time_span = pd.DataFrame(
    {
        'Date' : pd.date_range(
            start = pd.to_datetime(lse['Date'][len(lse)-1]),
            end   = pd.to_datetime(lse['Date'][0]),
            freq  = 'D'
        )
    }
)
df = pd.merge(time_span, lse, on = 'Date', how = 'left')

# IMPORT TWEETS ID W/ PROBABILITIES
session    = Cluster().connect('twitter') ; session.execute('USE twitter')
query      = 'SELECT id, date, class FROM tweetsclass'
sentiments = session.execute(query, timeout = None)

sentiments['date'] = pd.to_datetime(sentiments['date'])

sentiments = sentiments.sort_values(by = 'date')
sentiments = sentiments.reset_index()

sentiments.drop(['Unnamed: 0', 'index'], axis = 1, inplace = True)

sentiments = sentiments.rename(
    columns = {
        'id'    : 'ID',
        'date'  : 'Date',
        'class' : 'Probability'
    }
)

# CLASSIFY TWEETS
sentclass = []

def sentclassifier(x):
    if   x > 0.7:
        return int( 1)
    elif x < 0.3:
        return int(-1)
    else:
        return int( 0)

for _ in range(len(sentiments['Probability'])):
    sentclass.append(sentclassifier(sentiments['Probability'][_]))

sentiments['Sentiments'] = sentclass ; del sentclass

# DATE : DAY / HOURS
Day  = [] ; Hour = []

def get_elemdate(x, elemdate):
    x = str(x)
    if elemdate == 'day':
        return x[  :10]
    elif elemdate == 'hour':
        return x[11:16]

for _ in range(len(sentiments['Date'])):
    Day.append( get_elemdate(sentiments['Date'][_], elemdate = 'day' ))
    Hour.append(get_elemdate(sentiments['Date'][_], elemdate = 'hour'))

sentiments['Day' ] = Day  ; del Day
sentiments['Hour'] = Hour ; del Hour

# EXPORT
sentiments.to_csv('sentiments_export.csv', encoding = 'utf-8')

# UNIQUE DAYS
Days = sentiments['Day']
Days = list(set(Days))

# MOOD SENTIMENT
Mood = []

for _ in Days:
    subs = sentiments[sentiments['Day'] == _]
    Mood.append(round(sum(subs['Sentiments'])/len(subs), 3))

mood = pd.DataFrame(
    {
        'Mood' : Mood,
        'Date' : Days
    }
)

# CHANGE LSE
lse['Change'] = None

for _ in range(len(lse)-1):
    lse['Change'][_] = round(((lse['Open'][_] - lse['Open'][_+1])/lse['Open'][_+1])*100 ,2)

# MERGE
df = pd.merge(time_span, lse,  on = 'Date', how = 'left')
df = pd.merge(df,        mood, on = 'Date', how = 'left')

# EXPORT
df.to_csv('changemood_export.csv', encoding = 'utf-8')

# DATA
date = df[ 'Date' ]
X    = df[ 'Mood' ].astype(float)
y    = df['Change'].astype(float)

# STANDARDIZE + REG
Xs = (X - np.nanmean(X, axis = 0)) /np.nanstd(X, axis = 0)
ys = (y - np.nanmean(y, axis = 0))/ np.nanstd(y, axis = 0)

reg = pd.DataFrame(
    {
        'Mood'   : Xs,
        'Change' : ys
    }
)
reg     = reg.dropna(axis = 0)
degrees = 3
polynom = np.poly1d(np.polyfit(reg['Mood'], reg['Change'], degrees))
xp      = np.linspace(-2.2, 2.1, 20)

# TIME SERIES
ts_x = pd.Series(Xs.values, index = df['Date']) ; ts_x = ts_x.rename("Mood")
ts_y = pd.Series(ys.values, index = df['Date']) ; ts_y = ts_y.rename("Change")

# PLOTS
plot = False

if plot:
    ts_x.plot(
        kind   = 'line',
        legend = True,
        grid   = True
    )
    ts_y.plot(
        legend = True,
        grid   = True
    )

if plot:
    axes = plt.axes()
    axes.grid()
    plt.xlabel('Mood')
    plt.ylabel('Change')
    plt.scatter(reg['Mood'], reg['Change'])
    plt.plot(xp, polynom(xp), c = 'r')
    plt.show()