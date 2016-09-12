#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pyspark import SparkContext, SparkConf

os.environ['SPARK_HOME'] = '/home/bigdata/spark-1.5.2'
sys.path.append("/home/bigdata/spark-1.5.2/python")

conf = SparkConf.setMaster("spark://61.147.114.89:7077").setAppName("pySparkTest")

sc = SparkContext(conf)

logFile = "/home/tom/spark-1.6.0/README.md"
logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()

print("Lines with a: %i, lines with b: %i"%(numAs, numBs))