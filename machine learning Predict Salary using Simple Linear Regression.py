#!/usr/bin/env python
# coding: utf-8

# # objective :
# ##### We have to predict the salary of an employee given how many years of experience they have.

# In[1]:


import pandas as pd 
dataset=pd.read_csv("C:\\Users\\Vidya sagar\\salary.csv")


# In[5]:


dataset.head()


# In[6]:


# predictor i.e yearsExperience 
X=dataset.iloc[:,:-1].values
#dependent variable y i.e salary
y=dataset.iloc[:,1].values


# In[8]:


print(X)


# In[9]:


print(y)


# ### split dataset into training set and test set
# 

# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# #### fit simple linear Regression model to training set

# In[19]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# ##### predict the test set 

# In[20]:


y_pred=regressor.predict(X_test)


# In[21]:


print(y_pred)


# In[22]:


y_test


# In[25]:


import numpy as np
from math import sqrt
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[26]:


rmse(y_pred,y_test)


# In[31]:


regressor.score(X_train,y_train)


# In[37]:


# Visualize training set results
import matplotlib.pyplot as plt
# plot the actual data points of training set
plt.scatter(X_train, y_train, color = 'red')
# plot the regression line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[38]:


# Visualize test set results
import matplotlib.pyplot as plt
# plot the actual data points of test set
plt.scatter(X_test, y_test, color = 'red')
# plot the regression line (same as above)
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# ###### Make new predictions

# In[41]:


new_salary_pred = regressor.predict([[15]])
print(new_salary_pred)


# In[40]:


import statsmodels.formula.api as smf

# Modeling the effect of "height", weight and "gender" on "index"
model1 = smf.ols("Salary ~ YearsExperience", data = dataset).fit()
print(model1.summary())


# In[ ]:




