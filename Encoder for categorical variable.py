#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd 
x=pd.read_csv("C:\\Users\\Vidya sagar\\Desktop\\mutliregression_data.csv")


# In[25]:


x.head()


# In[30]:


x.shape


# In[28]:


#create categorical variable for Pclass

x= pd.get_dummies(x, columns=["Country"])
x.head()


# In[14]:


print(x)


# In[ ]:




