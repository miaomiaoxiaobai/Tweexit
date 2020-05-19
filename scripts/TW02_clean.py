#!/usr/bin/python

"""
	Clean and format streamed tweets
"""

import os
from pyspark     import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from os          import listdir

os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.ui.port=4040 --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.0,com.datastax.spark:spark-cassandra-connector_2.11:2.0.0-M3 pyspark-shell'

conf       = SparkConf().set("spark.cassandra.connection.host", "127.0.0.1")
sc         = SparkContext(conf=conf) 
sqlContext = SQLContext(sc)

# list files in dir
filelist = listdir("/home/guest/raw")

for file_ in filelist :
    dfs    = sqlContext.read.json('/home/guest/raw/' + file_)
    testdf = dfs.select('created_at', 'id','text')
    testdf = testdf.selectExpr('id as id', "created_at as date", 'text as text')
    testdf.write.option("sep","|").format("org.apache.spark.sql.cassandra").mode('append').options(table = "tweetsclean", keyspace = "twitter").save()