#!/usr/bin/env python
# coding: utf-8

# In[94]:


#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
get_ipython().run_line_magic('matplotlib', 'inline')


# In[95]:


#Perform Exploratory Data Analysis in just one line of code
pandas_profiling.ProfileReport(pd.read_csv('Titanic_Dataset.csv'))


# In[96]:


#Import Titanic Dataset
titanic_data = pd.read_csv('Titanic_Dataset.csv')


# In[97]:


#Count number of passenger who survived and NOT survived: 0 = Not Survived, 1 = Survived
sns.countplot(x='Survived',data=titanic_data)


# In[98]:


#Count number of males and females who survived and NOT survived: 0 = Not Survived, 1 = Survived
sns.countplot(x='Survived',hue='Sex',data=titanic_data)


# In[99]:


#Count which class passengers survived most
sns.barplot(x='Pclass', y='Survived', data=titanic_data)


# In[100]:


#Verify missing values in each column
titanic_data.isnull().sum()


# In[101]:


#Check data types of each column and hence finding out which columns are categorical in nature
titanic_data.info()


# In[102]:


#Visualize and find out average age by class
plt.figure(figsize=(12, 8))
sns.boxplot(x='Pclass',y='Age',data=titanic_data,palette='winter')


# In[103]:


#Function to impute missing values in age column based on class
def imput_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        return int(titanic_data[titanic_data["Pclass"] == Pclass]["Age"].mean())
    else:
        return Age


# In[104]:


#Impute missing values in age column by average age
titanic_data["Age"] = titanic_data[["Age", "Pclass"]].apply(imput_age,axis=1)


# In[105]:


#Drop Cabin column
titanic_data.drop("Cabin",inplace=True,axis=1)


# In[106]:


#Impute Embarked missing values with most common values
most_common_value = 'S'

for data in titanic_data:
    titanic_data['Embarked'] = titanic_data['Embarked'].fillna(most_common_value)


# In[107]:


#Check if there are any missing values left
titanic_data.isnull().sum()


# In[108]:


#Convert categorical variables into Dummy variables
sex = pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
pclass = pd.get_dummies(titanic_data["Pclass"],drop_first=True)


# In[109]:


#Drop irrelevant columns
titanic_data.drop(['PassengerId','Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)


# In[110]:


#Validate the columns and data
titanic_data.head()


# In[111]:


#Concatenate the dummy variables, created above
titanic_data = pd.concat([titanic_data,sex,embark,pclass],axis=1)
titanic_data.head()


# In[112]:


#Create Feature variable X and Target variable y
X = titanic_data.drop("Survived",axis=1)
y = titanic_data['Survived']


# In[113]:


#Split the data into training set (70%) and test set (30%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)


# In[114]:


#Fit the logistic Regression Model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(random_state=50)
logmodel.fit(X_train,y_train)

#Predict the value for new, unseen data
pred = logmodel.predict(X_test)
pred


# In[115]:


#Check the values of first record correspoding to the first vector value given above.
X_test.head(1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:




