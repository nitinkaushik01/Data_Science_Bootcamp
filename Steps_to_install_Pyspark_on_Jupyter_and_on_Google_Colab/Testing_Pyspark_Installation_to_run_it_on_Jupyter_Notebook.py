#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Testing pyspark installation
import findspark
findspark.init()
findspark.find()
import pyspark
findspark.find()


# In[3]:


#Initiate Spark Context
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
conf = pyspark.SparkConf().setAppName('SparkApp').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)


# In[4]:


#Example Test Code
numeric_val = sc.parallelize([1,2,3,4])
numeric_val.map(lambda x: x*x*x).collect()


# In[5]:


#Stop the Session
sc.stop()


# In[ ]:




