#!/usr/bin/python

"""
    Spark Streaming used with the Twitter API to continuously collect
    tweets filtered by keyword 'brexit' and store them in a cassandra db
"""

# LIBRAIRIES
import tweepy
import socket
import codecs, json
import io
import os
import csv
import datetime
import time
import json
from tweepy            import OAuthHandler
from tweepy            import Stream
from tweepy.streaming  import StreamListener
from cassandra.cluster import Cluster

# LOGIN PARAMETERS
with open('login.json', 'r') as f:
    login_parameters = json.load(f)

consumer_key        = login_parameters.get("consumer_key")
consumer_secret     = login_parameters.get("consumer_secret")
access_token        = login_parameters.get("access_token")
access_token_secret = login_parameters.get("access_token_secret")

# AUTHENTIFICATION PROCESS
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# AUTHENTIFICATION API
api = tweepy.API(auth)

# CASSANDRA DRIVER
script_cass = "cqlsh -f SCRIPTCASS.cqlsh"
os.system("bash -c '%s'" % script_cass)

# CLUSTER
cluster = Cluster()
session = cluster.connect('twitter') ; session.execute('USE twitter')

# TIMESTAMPS (FILENAMES)
ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d--%H-%M-%S')

# FILTER
keyword  = 'brexit'

# LISTENER
class StdOutListener(StreamListener):
    '''
        Handles data received from stream
    '''
    def on_data(self, data):
        jsonData = json.loads(data)
        # save to cassandra
        requete = "insert into twitter.tweetsrawtext (id, tweet) values(uuid(), $$"+data+"$$)"
        session.execute(requete)
        return True
    def on_error(self, status_code):
        # continue listening
        print('Error with status code: ' + str(status_code))
        return True 
    def on_timeout(self):
        # continue listening
        print('Timeout...')
        return True

if __name__ == '__main__':
    listener = StdOutListener()
    auth     = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream   = Stream(auth, listener)
    stream.filter(track = [keyword])

###