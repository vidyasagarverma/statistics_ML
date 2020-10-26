
# coding: utf-8

# Background:
# ![title](download.jpg)
# 
# On April 10, 1912 the RMS Titanic left Southhampton, England headed for New York. Aboard were 2,435 passengers and 892 crew members. Five days later, about 20 minutes before midnight, the Titanic hit an iceberg in the frigid waters about 375 miles south of New Foundland. Within approximately two and a half hours the ship had split apart and sunk, leaving just over 700 survivors.

# In[37]:

#import package 
import pandas as pd 
import numpy as np 
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)


# In[2]:

def concat_df(train_data,test_data):
    return pd.concat([train_data,test_data],sort=True).reset_index(drop=True)


# In[3]:

#import dataset 
train_data=pd.read_csv("C:\\Users\\Vidya sagar\\Downloads\\titanic\\train.csv")
#import test dataset 
test_data=pd.read_csv("C:\\Users\\Vidya sagar\\Downloads\\titanic\\test.csv")


# In[4]:

#checking data quality
print("train_data cotains  "+  str(len(train_data))+'rows and '+str(len(train_data.columns))+"columns")
print("test_data cotains  "  +  str(len(test_data))+'rows and '+str(len(test_data.columns))+"columns")


# In[5]:

print("display first value of training_data")
train_data.head()


# In[6]:

print("display first 5 value of test_data")
test_data.head()


# In[7]:

print("missing value in train_data")
train_data.isnull().sum()
print("missing value in test_data")
test_data.isnull().sum()


# In[8]:

print("missing value in test_data")
test_data.isnull().sum()


# In[9]:

#concat both data for data cleaning
df_all=concat_df(test_data,train_data)
print(df_all)


# In[10]:

#data cleaning 
#missing data of Age in dataset
print("Missing  for age in entire dataset "+ str(df_all['Age'].isnull().sum()))
#print missing data in percentage
print("Missing for age in percentage"+ str(round(df_all['Age'].isnull().sum()/len(df_all)*100))+'%')


# In[11]:

df = px.df_all['Age']
fig = px.histogram(df_all, x="age")

fig.show()


# In[ ]:

dfp=px.data.tips()
fig=px.histogram(df_all.groupby("Pclass")["Age"],x='Age')
fig.show()


# In[ ]:

train_data.columns


# In[ ]:

df_all.groupby("Age")['Pclass'].plot()


# In[ ]:




# In[ ]:


df_all.groupby('Pclass')['Age'].hist()


# In[ ]:


display(df_all.groupby('Pclass')['Age'])


# In[ ]:

print("Median of age of diffferent pclass")
display(train_data.groupby('Pclass')["Age"].median())
print("Median of age of diffferent pclass and sex")
display(train_data.groupby(['Pclass','Sex'])["Age"].median())
print("Number of cases ")
display(train_data.groupby(['Pclass','Sex'])["Age"].count())


# In[ ]:

#replace the missing value with the median of each group 
df_all['Age']=df_all.groupby(["Pclass",'Sex'])["Age"].apply(lambda x: x.fillna(x.median()))
df_all["Age"]


# In[ ]:

#missing data of Fare in dataset
print("Missing  for Fare in entire dataset "+ str(df_all['Fare'].isnull().sum()))


# In[ ]:

df_all[pd.isnull(df_all.Fare)]


# In[ ]:




# In[14]:

df_all.loc[df_all['Fare'].isnull()]


# In[15]:

#replacing missing value of Fare with median
Mr_Thomas=df_all.loc[(df_all['Pclass']==3) & (df_all['SibSp']==0) & (df_all['Embarked']=="S")]["Fare"].median()
print(Mr_Thomas)


# # cabin 
# clearing cabin missing data 

# In[16]:

display(train_data['Cabin'].unique())
#number of different cabin and missing cases
print("There are " + str(train_data['Cabin'].nunique())+" diffent cabins" + "and "  +  "missing cases"+str(train_data["Cabin"].isnull().sum()))


# In[17]:

# proportion of "cabin" missing
round(687/len(df_all["PassengerId"]),4)#


# 
# 
# 52 % of records are missing, which means that imputing information and using this variable for prediction is probably not wise. We'll ignore this variable in our model.

# In[26]:

#drop this variable from model
df_all.drop('Cabin', axis=1, inplace=True)


# # Embarked 

# In[18]:

#missing value in Embarked 
#df_all[pd.isnull(df_all['Embarked'])]
df_all['Embarked'].isnull().sum()
df_all.loc[df_all['Embarked'].isnull()]


# In[ ]:




# In[20]:

df_all.loc[(df_all['Pclass']==1) & (df_all["Fare"]<=80)]["Embarked"].value_counts()


# In[21]:

df_all.columns


# In[22]:

df_all['Embarked'].hist()


# In[23]:

df_all.loc[df_all["Embarked"].isnull()]="S"


# In[24]:

df_all["Embarked"].isnull().sum()


# # Exploratory Data Analysis ¶

# In[32]:

#Exploration of Age 
df = px.data.tips()
px.plot(df_all, x='Age', y='Survived')


# In[38]:



sns.kdeplot(df_all["Age"][df_all.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(df_all["Age"][df_all.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
plt.show()



# The age distribution for survivors and death is actually very similar. One notable difference is that, of the survivors, a larger proportion were children. The passengers evidently made an attempt to save children by giving them a place on the life rafts.

# # Exploration of Fare

# In[39]:

plt.figure(figsize=(15,8))
sns.kdeplot(df_all["Fare"][df_all.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(df_all["Fare"][df_all.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
# limit x axis to zoom on most information. there are a few outliers in fare. 
plt.xlim(-20,200)
plt.show()


# As the distributions are clearly different for the fares of survivors vs. deceased, it's likely that this would be a significant predictor in our final model. Passengers who paid lower fare appear to have been less likely to survive. 

# # Exploration of Passenger Class

# In[43]:

sns.barplot('Pclass', 'Survived', data=train_data)
plt.show()


# Unsurprisingly, being a first class passenger was safes

# # Exploration of Embarked Port 

# In[45]:

sns.barplot('Embarked', 'Survived', data=train_data)
plt.show()


# Passengers who boarded in Cherbourg, France, appear to have the highest survival rate

# # Exploration of Traveling Alone vs. With Family

# In[47]:

## Create categorical variable for traveling alone

train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)


# In[51]:

sns.barplot('TravelAlone', 'Survived', data=train_data, color="mediumturquoise")
plt.show()


# Individuals traveling without family were more likely to die in the disaster than those with family aboard. Given the era, it's likely that individuals traveling alone were likely male.

# # Exploration of Gender Variable

# In[53]:

sns.barplot('Sex', 'Survived', data=train_data, color="mediumturquoise")
plt.show()


# This is a very obvious difference. Clearly being female greatly increased your chances of survival.
# 
# 

# In[ ]:



