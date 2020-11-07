#!/usr/bin/env python
# coding: utf-8

# ## Handling with categorical variable : 
# why do we need to handle ?: Machine learning  models like regression,logistic  require all input and output variables to be numeric.
# This means that if your data contains categorical data, you must encode it to numbers before you can fit and evaluate a model.
# 
# 

# ## Approach for dealing with categorical Data : 
# There are three common approaches for converting ordinal and categorical variables to numerical values. They are:
# 
# Label Encoding
# 
# One-Hot Encoding
# 
# 

# #### Approach 1: label Encoding: 
# If the variable has a clear ordering, then that variable would be an ordinal variable. Examples of ordinal variables include: socio economic status (“low income”,”middle income”,”high income”), education level (“high school”,”BS”,”MS”,”PhD”), income level (“less than 50K”, “50K-100K”, “over 100K”), satisfaction rating (“extremely dislike”, “dislike”, “neutral”, “like”, “extremely like”). This approach work well when the data is ordinal 

# In[3]:


# lable encoding with python 
# example of a ordinal encoding
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
# define data
Breakfast = asarray([['Never'], ['Rarely'], ['Mostly day'],['Daily']])
print(Breakfast)
# define label encoding
encoder = LabelEncoder()
# transform data
result = encoder.fit_transform(Breakfast)
print(result)


# #### problem with Label Encoding : 
# 
# 
# This approach assumes an ordering of the categories: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).
# 
# This assumption makes sense in this example, because there is an indisputable ranking to the categories. Not all categorical variables have a clear ordering in the values
# 
# Lable Encoding works good with ordinal dataset as label that ,we provide to it, is  comparable.This approach is not good when data is not ordinal or when data is  nomial data
# 
# ###### Nominal
# A nominal scale describes a variable with categories that do not have a natural order or ranking. You can code nominal variables with numbers if you want, but the order is arbitrary and any calculations, such as computing a mean, median, or standard deviation, would be meaningless.
# 
# Examples of nominal variables include:
# 
# color, country ,genotype, blood type, zip code, gender, race, eye color, political party. 
# let's look at problem while applying to nomial dataset 

# In[6]:


#  labelEncoding with nomial dataset 
# example of a label encoding
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
# define data
Country = asarray([['America'], ['India'], ['China'],['Pakistan']])
print(Country)
# define Label encoding
encoder = LabelEncoder()
# transform data
result = encoder.fit_transform(Country)
print(result)


#  For categorical variables, it imposes an ordinal relationship where no such relationship may exist like above example America(0)<China(1) which doesn't make any sense here.  so ,This can cause problems and a one-hot encoding may be used instead.

# #### Approach 2 : OneHot Encoding :
# For categorical variables where no ordinal relationship exists, the Label encoding may not be enough, at best, or misleading to the model at worst.
# 
# Forcing an ordinal relationship via an ordinal encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results (predictions halfway between categories).
# 
# In this case, a one-hot encoding can be applied to the ordinal representation.One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data. 
# To understand this, we'll work through an example.

# ![image.png](attachment:image.png)

# In[10]:


# example of a one hot encoding
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
# define data
data = asarray([['red'],['red'],['yellow'] ,['green'], ['yellow']])
print(data)
# define one hot encoding
encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(data)
print(onehot)


# In[21]:



#Another Approach :  get_dummies()method 
import pandas as pd 
import numpy as np 
li=["red","red","yellow","green","yellow"]
print(pd.get_dummies(li))


# In[17]:


import pandas as pd 
import numpy as np

dataset=pd.read_csv("https://raw.githubusercontent.com/vidyasagarverma/statistics_ML/master/startup50.csv")


# In[18]:


dataset.head()


# In[19]:


state=dataset['State']


# In[20]:


pd.get_dummies(state)


# In[ ]:




