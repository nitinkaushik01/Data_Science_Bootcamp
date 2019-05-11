
# coding: utf-8

# In[76]:

from IPython.display import Image
Image(filename="Downloads/EDA_Diagram.png")


# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')


# # Univariate Analysis

# ## Categorical Variables

# In[38]:

df = pd.read_csv('iris.csv')
df.shape
df.head()


# In[3]:

df.groupby(['Species']).count()


# In[4]:

freq_table = df.groupby(['Species']).size().reset_index(name='Count').rename(columns={'Sepcies':'Species'})
freq_table


# In[5]:

plt.bar(freq_table['Species'], freq_table['Count'])
plt.show()


# In[6]:

freq_table['Count%'] = freq_table['Count']/sum(freq_table['Count'])*100
freq_table


# ## Numerical Variables

# In[7]:

df.groupby(['SepalLengthCm']).size().reset_index(name='Count').rename(columns={'SepalLengthCm':'SepalLengthCm'})


# In[9]:

plt.figure(figsize = (10,7))
x = df["SepalLengthCm"]
plt.hist(x, bins=20, color="green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal Length cm")
plt.ylabel("Count")


# In[10]:

new_df = df[["SepalLengthCm","SepalWidthCm", "PetalLengthCm","PetalWidthCm"]]
new_df.head()


# In[11]:

plt.figure(figsize = (10,7))
new_df.boxplot()


# In[16]:

df["SepalLengthCm"].median()


# In[17]:

np.percentile(df.SepalLengthCm, 25)


# In[18]:

np.percentile(df.SepalLengthCm, 75)


# ## Binning or Discretization

# In[39]:

sepal_length_ranges = ["[{0} - {1})".format(SepalLengthCm, SepalLengthCm + 1.0) for SepalLengthCm in range(4, 8, 1)]
sepal_length_ranges


# In[40]:

count_sepal_length_ranges = len(sepal_length_ranges)
count_sepal_length_ranges


# In[41]:

df['SepalLengthCm_Range'] = pd.cut(x=df['SepalLengthCm'], bins=count_sepal_length_ranges, labels=sepal_length_ranges)


# In[42]:

df.head()


# In[43]:

sepal_len_hist_df = df['SepalLengthCm_Range'].value_counts()
sepal_len_hist_df


# In[44]:

df_range_hist = pd.DataFrame(sepal_len_hist_df).reset_index()
df_range_hist.columns = ['SepalLengthCm_Range', 'Count']
df_range_hist


# In[53]:

plt.bar(df_range_hist["SepalLengthCm_Range"], df_range_hist["Count"])
plt.show()


# # Bi-Variate Analysis

# ## Numerical-Numerical Variable Analysis

# In[56]:

#Scatter Plot
sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', fit_reg=False, data=df)


# In[57]:

#Scatter Plot Matrix
sns.pairplot(hue='Species', data=df)


# In[58]:

#Correlation 
df.corr()


# ## Categorical-Categorical Variable Analysis

# In[63]:

df2 = pd.read_csv('iris_with_colors.csv')
df_plot = df2.groupby(['Color', 'Species']).size().reset_index().pivot(columns='Color', index='Species', values=0)
df_plot.plot(x=df_plot.index, kind='bar', stacked=True)


# In[64]:

pd.crosstab(df2.Species,df2.Color)


# In[65]:

df_mean = df.groupby(['Species']).mean().reset_index()
df_mean


# In[66]:

plt.plot(df_mean['Species'], df_mean['SepalLengthCm'])


# In[68]:

df_mean = df.groupby(['Species']).mean().reset_index()
df_var = df.groupby(['Species']).var().reset_index()
z = (5.006 - 5.936)/np.sqrt(np.square(0.1242)/50 + np.square(0.266433)/50)
z


# ## Numerical-Categorical Variable Analysis

# In[70]:

df_mean = df.groupby(['Species']).mean().reset_index().drop(['Id'], axis=1).add_prefix('mean_')
df_var = df.groupby(['Species']).var().reset_index().drop(['Id','Species'], axis=1).add_prefix('var_')
df_count = df.groupby(['Species']).count().reset_index().drop(['Id', 'Species'], axis=1).add_prefix('count_')
pd.concat([df_mean, df_var, df_count], axis=1)


# In[71]:

k=3
N = 50+50+50
Numerator = np.sum([np.square(5.006-5.843)/(k-1), np.square(5.936-5.843)/(k-1), np.square(6.588-5.843)/(k-1)])
Numerator


# In[72]:

Setosa = np.sum(np.square(df[df['Species'] == 'Iris-setosa'].SepalLengthCm - 5.006)/(N-k))
Versicolor = np.sum(np.square(df[df['Species'] == 'Iris-versicolor'].SepalLengthCm - 5.936)/(N-k))
Verginica = np.sum(np.square(df[df['Species'] == 'Iris-virginica'].SepalLengthCm - 6.588)/(N-k))
Denominator = Setosa+Versicolor+Verginica
Denominator


# In[73]:

F = Numerator/Denominator
F


# In[ ]:

#Link to Calculate F-Statistics
http://onlinestatbook.com/2/calculators/F_dist.html


# In[75]:




# In[ ]:



