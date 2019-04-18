
# coding: utf-8

# ## Import the libraries

# In[7]:

import pymongo
from pymongo import MongoClient


# ## Establish connection

# In[22]:

connection = MongoClient('localhost', 27017)
db = connection.EmployeeDB


# ## Extract database and collection details

# In[23]:

data = db.employees
EmployeeList = data.find()


# ## Print each individual record

# In[25]:

for item in EmployeeList:
    print("Name: " + item["name"] + " " + "Dept: " + item["dept"])


# ## Create a dataframe to give structure to the NoSQL data

# In[26]:

import pandas as pd
df = pd.DataFrame(list(data.find()))
df


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



