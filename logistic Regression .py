#!/usr/bin/env python
# coding: utf-8

# 
# '''One of the task for data scientist working in  an e-commerce company is to know the potential of the customers to buy a product.One of the technique to know it is direct contact to customer and ask whether they are like to buy a product or not. But it is not very fancy method in this period. As being a data scientist , we need to solve this problem using our skill of machine learning algorithm.Since problem i.e whether customers will buy a product( =1) or not( =0).
# 
# Instead of trying to predict exactly whether the people will buy a product or not, you calculate the probability or a likelihood of the person saying yes. Basically you try to fit in probabilities between 0 and 1, which are the two possible outcomes. You also decide a cut off value/threshold and then conclude that people with a probability higher than the threshold will buy the product and vice versa.
# 
# This method is used by company to target potential customers. In this method, probability of people who are more likely to buy a product is calcaluted and  focus only on the customers who are most likely to say Yes.'''
# 
# 
# Logistic Regression 
# 
# Logistic Regression is technique borrowed by machine learning from the field of statistics where the dependent variable is categorical and not continuous. It predicts the probability of the outcome variable.
# 
# Logistic regression is named for the function used at the core of the method, the logistic function.The logistic function, also called the sigmoid function was developed by statisticians
# 
# ![image.png](attachment:image.png)
# 
# sigmoid function is  an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.
# 
#                      1 / (1 + e^-value)
#  where e is expotential function 
#  
#  ###### representation of logistic Regression 
# just like a lInear regression model, a logistic regression model also computes a weighted sum of input features(and adds a bias term to it)i.e output = b0 + b1*x1 +... where b0,b1 and b2 .. are the coefficient of model and need to discover the best values for the coefficients. 
# However, unlike linear regression, it calculates the logistic of the results so that the output is always between 0 and 1.using maximum likelihood to fit a logistic regression model .The interpretation of the weights in logistic regression differs from the interpretation of the weights in linear regression, since the outcome in logistic regression is a probability between 0 and 1.In a linear regression model, b1 gives the average change in Y associated with a one-unit increase in X. In contrast,
# in a logistic regression model, increasing X by one unit changes the log oddsby b1 , or equivalently it multiplies the odds by e^b1

# In[17]:


'''Aim : Using  iphone dataset we have to predict t if the customer will purchase an iPhone or not given their gender
, age and salary'''
#import the dataset 
import pandas as pd 

dataset=pd.read_csv("https://raw.githubusercontent.com/vidyasagarverma/statistics_ML/master/iphone_sell.csv")

dataset.head()


# In[2]:


'''define  X as independent variable in this case Gender , Age and Salary and dependent variable y as purchase Iphone'''
X=data.iloc[:,:-1].values
y=data.iloc[:,3].values


# In[52]:


#Transform Gender into number 
from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
X[:,0]=labelEncoder.fit_transform(X[:,0])

#Split Data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.25, random_state=0)
#for using logistic Regression, we need to feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test) 
# Fit logistic regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver="liblinear")
classifier.fit(X_train, y_train)
#make prediction
y_pred = classifier.predict(X_test)
print(y_pred)


# In[29]:


from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred)
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred)
print("Recall score:",recall)


# In[60]:


#make a new prediction 

#Male aged 21 making $40,000, firstly transform the input 
sc_X_val = sc.transform(np.array([[0,21,40000]]))

y_pred = classifier.predict(sc_X_val)

print(y_pred)

