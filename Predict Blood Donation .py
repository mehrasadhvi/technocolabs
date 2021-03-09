#!/usr/bin/env python
# coding: utf-8

# Loading the blood donations data

# In[159]:


import seaborn as sns
import random
random.seed(100)
import pandas as pd
import sklearn
data=pd.read_csv('C:/Users/sadhv/OneDrive/Desktop/technocolabs/transfusion.csv')
data.head()


# Inspecting transfusion DataFrame

# In[160]:


data.info()


# In[161]:


data.describe()


# In[162]:


sns.boxplot(y=data['Monetary (c.c. blood)'])


# In[163]:


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


# In[164]:


uw=inqr(data['Time (months)'])
data = data[data['Time (months)']< uw]
sns.boxplot(y=data['Time (months)'])


# Creating target column

# In[165]:


target=data['whether he/she donated blood in March 2007']
data=data.drop(['whether he/she donated blood in March 2007'],axis=1)
target.value_counts()


# Checking target incidence

# In[166]:


target.value_counts(normalize=True)*100


# Splitting transfusion into train and test datasets
# 

# In[167]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data,target,stratify=target,test_size=0.3,random_state=42)
Y_train.value_counts(normalize=True)*100


# In[168]:


Y_test.value_counts(normalize=True)*100


# Selecting model using TPOT

# In[171]:


from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score
model=TPOTClassifier(generations=5,population_size=20,scoring='roc_auc',verbosity=2,cv=5)
model.fit(X_train,Y_train.values.ravel())
model.export('BloodTransfusion.py')


# In[172]:


from sklearn.metrics import accuracy_score
Y_pred=model.predict(X_test)
print("\nAUC score:%f"%(roc_auc_score(Y_test,Y_pred)*100))
print("\nAccuracy score:%f"%(accuracy_score(Y_test,Y_pred)*100))


# Checking the variance

# In[173]:


data.var()


#  Log normalization

# In[174]:


import numpy as np
data['LogNormalMonetary']=np.log(data['Monetary (c.c. blood)'])
data=data.drop(['Monetary (c.c. blood)'],axis=1)
data.head()


# In[175]:


data.var()


# Training the logistic regression model

# In[200]:


X_train,X_test,Y_train,Y_test=train_test_split(data,target,test_size=0.3,stratify=target)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train.values.ravel())
Y_pred=lr.predict(X_test)
Y_pred


# In[201]:


print(Y_test.values)


# Conclusion

# In[202]:


from sklearn.metrics import confusion_matrix
print("\nAUC score:%f"%(roc_auc_score(Y_test,Y_pred)*100))
print("\naccuracy score:%f"%(accuracy_score(Y_test,Y_pred)*100))
print(confusion_matrix(Y_test,Y_pred))

