#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression using Scikit Learn

# ## Multiple Linear Regression Equation

# ### y = a0 + a1*X 1 + a2*X2 + a3*X3 + a4*X4 +a5*X5

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd


# In[2]:


#Read Restaurant Profit Dara .csv file and divide the data into dependent and independent variables.
data = pd.read_csv('Restaurant_Profit_Data.csv')
data.head()


# In[3]:


#Check the shape of the data
data.shape


# In[4]:


#Create Feature Matrix and Dependent Variable Vector
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values


# In[5]:


#Didplay value of Feature Matrix
X


# In[6]:


#Display value of Dependent Variable vector
y


# In[7]:


# One Hot Encoding of categorical column called City
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lblencoder = LabelEncoder()
X[:, 3] = lblencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


# In[8]:


#Display feature matrix once the one hot encoding is done
X.astype(int)


# In[9]:


# Get rid of the Dummy Variable (Column called Chicago) 
X = X[:, 1:]


# In[11]:


#Diaplay the value of feature matrix after dummy variable is removed
X.astype(int)


# In[13]:


#Split the data into training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[14]:


len(X_train)


# In[16]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(X_train, y_train)


# In[24]:


#Print the constant and coeffcients
print(f'constant = {MLR.intercept_}')
print(f'coefficients = {MLR.coef_}')


# In[25]:


# Predicting the Test set results
y_predict = MLR.predict(X_test)


# In[26]:


# Display actual and predicted values side by side
df = pd.DataFrame(data=y_test, columns=['y_test'])
df['y_predict'] = y_predict
df


# In[27]:


# Predicting the sigle observation results. Here 1,0 represents that the city is Mumbai
singl_obs = [1,0,160349,134321,401400]
feature_array = np.array(singl_obs)
feature_array = feature_array.reshape(1, -1)
y_pred_single_obs = MLR.predict(feature_array)
round(float(y_pred_single_obs), 2)


# In[28]:


#Model Evaluation using R-Square
from sklearn import metrics
r_square = metrics.r2_score(y_test, y_predict)
print('R-Square Error:', r_square)


# In[29]:


# We have 5 column in Multiple Linear Regression Equation : Profit = a0+a1*Miscellaneous_Expenses+a2*Food_Innovation_Spend+
# a3*Advertising+a4*Mumbai+a5*Tokyo
#Model Evaluation using Adjusted R-Square. 
# Here n = no. of observations and p = no. of independent variables
n = 50
p = 5
Adj_r_square = 1-(1-r_square)*(n-1)/(n-p-1)
print('Adjusted R-Square Error:', Adj_r_square)


# In[ ]:




