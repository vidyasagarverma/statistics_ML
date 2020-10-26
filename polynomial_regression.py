#!/usr/bin/env python
# coding: utf-8

# ###### objective :
# 
# In this project we will try to predict salary of employee using salary_position dataset "t.ly/GSXx" . linear Regression and Polynomial Regression Algorithm we will use to predict salary and also try to compare both the model .
# 

# In[3]:


import pandas as pd 
dataset=pd.read_csv("https://raw.githubusercontent.com/vidyasagarverma/statistics_ML/master/position_salary.csv")


# In[13]:


dataset


# In[12]:


#As we have to predict Salary on the basis of Position and level, so salary is dependent variable and level is independent varibles
#define X as in independent variable and y as dependent 
#using pandas X and y are
X = dataset.iloc[:, 1:2].values
y=dataset.iloc[:,2].values


# In[7]:


# firstly we fit the linear model and make prediction using linear model
from sklearn.linear_model import LinearRegression
linear_regression=LinearRegression()
linear_regression.fit(X,y)


# In[8]:


#lets visualise Linear Regression Results using matplotlib 
import matplotlib.pyplot as plt 
plt.scatter(X,y)
plt.plot(X, linear_regression.predict(X))
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# In[10]:


#make prediction 
linear_regression.predict([[6.5]])


# prediction using this model seems to  be bad as a person having level 6.5 should be salary in between 15k to 20k but linear Regression model predict it 33k and  visualation of linear regression represent that linear Regression Alogrithm is not suitable for this dataset. so we further try any other dataset so that that will fit to the given dataset and provide the better result.In this Exercise we will try Polynomial Regession Model 

# ###### polynomial Regression 
# for using polymial Regression , Firstly we need to convert X into X_polynominal . For this very purpose we will use PolynomialFeatures from sklearn.preprocessing 

# In[15]:


from sklearn.preprocessing import PolynomialFeatures
poly_regression=PolynomialFeatures(degree=2)
X_polynomial=poly_regression.fit_transform(X)


# In[16]:


X_polynomial


# In[17]:


# Passing X_poly to LinearRegression
linear_regression2=LinearRegression()
linear_regression2.fit(X_polynomial,y)


# In[20]:


#Visualize Poly Regression Results
plt.scatter(X,y, color="red")
plt.plot(X, linear_regression2.predict(poly_regression.fit_transform(X)))
plt.title("Poly Regression Degree 2")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# From  the given plot we can see that salary having grade 6.5 is being predicted 190k, which is far away from the actual value so we fit another degree of polynomial i.e. polynomial of 3rd degree

# In[23]:


## make prediction 
linear_regression2.predict(poly_regression.fit_transform([[6.5]]))


# In[24]:


# fit a model of 3rd polynomial 
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X_polynomial=poly_reg.fit_transform(X)


# In[25]:


# Passing X_poly to LinearRegression
linear_regression3=LinearRegression()
linear_regression3.fit(X_polynomial,y)


# In[26]:


#visualation 
plt.scatter(X,y)
plt.plot(X,linear_regression3.predict(poly_reg.fit_transform(X)))
plt.title(" 3rd degree polynomialRegression fit ")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# In[27]:


#make predication 
linear_regression3.predict(poly_reg.fit_transform([[6.5]]))


# still we have not got the reasonable fit of the model as prediction is not good  so we further move on to 4th degree polynomial.we will repeat the same step again 

# In[28]:


from sklearn.preprocessing import PolynomialFeatures
poly_regr=PolynomialFeatures(degree=4)
X_polynomial=poly_regr.fit_transform(X)


# In[30]:


# Passing X_poly to LinearRegression
linear_regression4=LinearRegression()
linear_regression4.fit(X_polynomial,y)


# In[32]:


#visualation 
plt.scatter(X,y)
plt.plot(X,linear_regression4.predict(poly_regr.fit_transform(X)))
plt.title(" 4th degree polynomialRegression fit ")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# In[33]:


linear_regression4.predict(poly_regr.fit_transform([[6.5]]))


# wow polynomial of 4th degree seems to be reasonable as this model has predict a person having grade 6.5 is 15k that we supposed to be the salary of person of grade 6.5

# In[ ]:




