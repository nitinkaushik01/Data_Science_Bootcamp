
# coding: utf-8

# ## Import Libraries

# In[87]:

import bs4 as bs
import urllib.request


# ## Store HTML source code in a variable

# In[88]:

src = urllib.request.urlopen('https://www.nytimes.com/').read()


# In[89]:

bsoup = bs.BeautifulSoup(src, 'lxml')


# In[ ]:

print(bsoup)


# ## Extract Title of the web page

# In[92]:

print(bsoup.title.text)


# ## Extract URLs

# In[ ]:

for link in bsoup.find_all('a'):
    print(link.get('href'))


# ## Extract Paragraphs

# In[94]:

print(bsoup.p)


# In[95]:

print(bsoup.find_all('p'))


# In[98]:

print(bsoup.find('p').get_text())


# In[99]:

ptags = bsoup.find_all('p')
for p in ptags:
    print(p.text)


# ## Extract Table Data

# In[100]:

src = urllib.request.urlopen('http://www.espn.com/nba/statistics/player/_/stat/assists/sort/avgAssists/').read()
bsoup = bs.BeautifulSoup(src, 'lxml')
tbl = bsoup.find('table')


# In[101]:

tbl_rows = tbl.find_all('tr')
for tr in tbl_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    print(row)


# In[102]:

type(row)


# ## Extract Table Data using Pandas

# In[103]:

import pandas as pd 
data = pd.read_html("http://www.espn.com/nba/statistics/player/_/stat/assists/sort/avgAssists/")
for df in data:
    print(df)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



