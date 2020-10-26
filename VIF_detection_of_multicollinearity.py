#!/usr/bin/env python
# coding: utf-8

# In[17]:


import jovian


# ### Detecting Multicollinearity using VIF
# Let’s try detecting multicollinearity in a dataset to give you a flavor of what can go wrong.
# 
# I have created a dataset determining the salary of a person in a company based on the following features:
# 
# Gender (0 – female, 1- male)
# Age
# Years of service (Years spent working in the company)
# Education level (0 – no formal education, 1 – under-graduation, 2 – post-graduation)
# 

# ##### Dataset 
# The dataset used in the example below, contains the height, weight, gender and Body Mass Index for 500 persons. Here the dependent variable is Index

# In[18]:


# the dataset 
import pandas as pd
data = pd.read_csv('C:\\Users\\Vidya sagar\\Downloads\\500_Person_Gender_Height_Weight_Index.csv') 
  
# printing first few rows 
print(data.head())


# In[19]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
  
# creating dummies for gender 
data['Gender'] = data['Gender'].map({'Male':0, 'Female':1}) 
import statsmodels.formula.api as smf

# Modeling the effect of "height", weight and "gender" on "index"
model1 = smf.ols("Index ~ Height + Gender +  Weight", data = data).fit()
print(model1.summary())


# In[20]:


# the independent variables set 
X = data[['Gender', 'Height', 'Weight']] 
  
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
  
print(vif_data)


# 
# this variance inflation factor tells us that the variance of the Height coefficient is inflated by a factor of11.4 because Height is highly correlated with at least one of the other predictors in the model.As we can see, height and weight have very high values of VIF, indicating that these two variables are highly correlated. This is expected as the height of a person does influence their weight. Hence, considering these two features together leads to a model with high multicollinearity.
# 

# ##### Fixing Multicollinearity
# One solution to dealing with multicollinearity is to remove some of the violating predictors from the model. If we review the pairwise correlations again:

# In[21]:


X = data.drop(['Height','Index'],axis=1)
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns 
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
  
print(vif_data)


# Aha — the remaining variance inflation factors are quite satisfactory! That is, it appears as if hardly any variance inflation remains

# In[22]:


# Modeling the effect of  weight and "gender" on "index"
model2 = smf.ols("Index ~ Weight + Gender ", data = data).fit()
print(model2.summary())
 


# In[23]:


jovian.commit(filename="VIF_detection_of_multicollinearity-Copy1")


# In[ ]:




