#!/usr/bin/python

"""
    Daily collects the opening value of the London Stock Exchange value 
    and send it in the cassandra database
"""

# LIBRAIRIES

import os
import cassandra
import subprocess
import pandas      as pd
import datetime    as dt
from   cassandra.cluster import Cluster
from   bs4               import BeautifulSoup
from   collections       import namedtuple

# GET OPENING VALUE

def get_opening():
    parser  = "html5lib"
    url     = "https://in.investing.com/equities/london-stock-exchange-historical-data"
    command = "curl --url " + url + " --user-agent Mozilla/5.0"
    page_h  = BeautifulSoup(
        subprocess.Popen(
            command,
            stdout = subprocess.PIPE,
            shell  = True
        ).communicate()[0],
        parser
    )
    table   = pd.read_html(
        io     = str(page_h.findAll("table", {"class":"genTbl closedTbl historicalTbl"})),
        header = 0
    )[0]
    opening = namedtuple("opening", ["date", "value"])
    opening = opening(
        dt.datetime.strptime(table['Date'][0],'%b %d, %Y').strftime('%d-%m-%Y'),
        table['Open'][0]
    )
    return opening

opening = get_opening() ; print("date  : ", opening.date, "\nvalue : ", opening.value, " £")

# SEND TO CASSANDRA

def sendtocassandra():    
    session = Cluster().connect('twitter') ; session.execute('USE twitter')
    today   = dt.datetime.now().strftime('%d-%m-%Y')
    query   = "select * from lse where date = '%s' ;" % today
    """
    fetch row with current date in cassandra:
        if exists : value already in cassandra
        if error  : put new value in cassandra
    """
    try:
        row = session.execute(query, timeout = None)[0]
        print('Current opening value already in cassandra')
    except IndexError:
        requete = "insert into twitter.lse (date, val) values('%s', %f)"\
            % (opening.date, opening.value)
        session.execute(requete, timeout = None)
        print('Opening value successfully inserted in cass')

sendtocassandra()

if __name__ == '__main__':
    opening = get_opening() 
    sendtocassandra()
    print(
        "date  : ", opening.date, "\nvalue : ", opening.value, " £")
# END