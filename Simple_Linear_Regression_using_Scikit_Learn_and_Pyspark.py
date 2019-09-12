
# coding: utf-8

# # Simple Linear Regression using Scikit Learn

# ### Simple Linear Regression Equation
# 
# 

# ### y = a0 + a1*X                                

# In[44]:

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[45]:

#Read Student Grades .csv file and divide the data into dependent and independent variables.
data = pd.read_csv('Student_Grades_Data.csv')
data.head()


# In[46]:

data.shape


# In[47]:

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[48]:

X


# In[49]:

y


# In[50]:

#Split the data into training and test datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[51]:

y_test


# In[52]:

#Fit the Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)


# In[53]:

#Print the 
print(f'a0 = {LinReg.intercept_}')
print(f'a1 = {LinReg.coef_}')


# In[54]:

#Predicted grade scores from test dataset
y_predict = LinReg.predict(X_test)
y_predict


# In[55]:

#Actual grade scores from test dataset
y_test


# In[56]:

#Grades Vs Time to Study visualization on Training Data
plt.scatter(X_train, y_train, color='Blue')
plt.plot(X_train, LinReg.predict(X_train), color='Black')
plt.title('Grades Vs Time to Study (On Training Data)')
plt.xlabel('Time to Study')
plt.ylabel('Grades')
plt.show()


# In[57]:

#Grades Vs Time to Study visualization on Test Data
plt.scatter(X_test, y_test, color='Red')
plt.plot(X_train, LinReg.predict(X_train), color='Black')
plt.title('Grades Vs Time to Study (On Test Data)')
plt.xlabel('Time to Study')
plt.ylabel('Grades')
plt.show()


# In[66]:

#Predicting Grade of a student when he studied for 10 Hrs. Example of how to pass an external value,
#Independent of Test or Training Dataset

Predict_Grade = LinReg.predict(10)
Predict_Grade


# In[59]:

#Model Evaluation using R-Square
from sklearn import metrics
r_square = metrics.r2_score(y_test, y_predict)
print('R-Square Error:', r_square)


# In[60]:

#For Illustration Purpose Only. 
#Considering Multiple Linear Equation with two Variables : grade = a0 + a1*time_to_study + a2*class_participation
#Model Evaluation using Adjusted R-Square. 
# Here n = no. of observations and p = no. of independent variables

n = 50
p = 2
Adj_r_square = 1-(1-r_square)*(n-1)/(n-p-1)
print('Adjusted R-Square Error:', Adj_r_square)


# In[61]:

#Model Evaluation using Mean Square Error (MSE)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))


# In[62]:

#Model Evaluation using Root Mean Square Error (RMSE)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))


# In[63]:

#Model Evaluation using Mean Absolute Error (MAE)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))


# In[ ]:



