#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import random
import pickle
random.seed(100)
import pandas as pd
data=pd.read_csv('C:/Users/sadhv/OneDrive/Desktop/technocolabs/transfusion.csv')
data.head()


# In[2]:


hp=[]
def inqr(hp):
    Q1=hp.quantile(0.25)
    Q3=hp.quantile(0.75)
    IQR=Q3-Q1
    Upper_Whisker = Q3+1.5*IQR
    return Upper_Whisker
uw=inqr(data['Monetary (c.c. blood)'])
data = data[data['Monetary (c.c. blood)']< uw]
sns.boxplot(y=data['Monetary (c.c. blood)'])


# In[3]:


uw=inqr(data['Time (months)'])
data = data[data['Time (months)']< uw]
sns.boxplot(y=data['Time (months)'])


# In[4]:


target=data['whether he/she donated blood in March 2007']
data=data.drop(['whether he/she donated blood in March 2007'],axis=1)
target.value_counts()


# In[5]:


import numpy as np
data['LogNormalMonetary']=np.log(data['Monetary (c.c. blood)'])
data=data.drop(['Monetary (c.c. blood)'],axis=1)
data.head()


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data,target,test_size=0.3,stratify=target)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train.values.ravel())
pickle.dump(lr,open('model.pkl','wb'))

