
# coding: utf-8

# ## Import Libraries

# In[41]:

import requests
import pandas as pd
from prettytable import PrettyTable
import json


# ## Create pretty table object

# In[42]:

tableobj = PrettyTable()


# ## Store API Key in a variable & concatenate it with API endpoint

# In[45]:

KeyVal = '37d0d5d1-eca3-4739-8723-8b4dce81be2b'

api_endpoint = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?CMC_PRO_API_KEY='
api_endpoint += KeyVal
api_endpoint


# ## Fetch data in JSON format

# In[47]:

json_data = requests.get(api_endpoint).json()
cryptodata = json_data['data']


# In[50]:

requests.get(api_endpoint).json()


# ## Store data in Pretty Table

# In[51]:

for currency in cryptodata:
    curr_name = currency['name']
    curr_price = currency['quote']['USD']['price']
    curr_change_1h = currency['quote']['USD']['percent_change_1h']
    curr_change_24h = currency['quote']['USD']['percent_change_24h']
    curr_change_7d = currency['quote']['USD']['percent_change_7d']
    tableobj.add_row([curr_name, curr_price, curr_change_1h, curr_change_24h, curr_change_7d])


# ## Create External Field names and assign as column names in Pretty table

# In[52]:

tableobj.field_names =  ["Currency Name", "Currency Price", "Currency 1h Change", "Currency 24h change", "Currency 7d change"]


# In[53]:

print(tableobj)


# ## Store data in Text File format

# In[54]:

table_txt = tableobj.get_string()
with open('output.txt', 'w') as file:
    file.write(table_txt)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



