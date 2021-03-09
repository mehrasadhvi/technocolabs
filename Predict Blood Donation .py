#!/usr/bin/env python
# coding: utf-8

# Loading the blood donations data

# In[44]:


import random
random.seed(100)
import pandas as pd
import sklearn
data=pd.read_csv('C:/Users/sadhv/OneDrive/Desktop/technocolabs/transfusion.csv')
data.head()


# Inspecting transfusion DataFrame

# In[14]:


data.info()


# In[15]:


data.describe()


# Creating target column

# In[16]:


target=data['whether he/she donated blood in March 2007']
data=data.drop(['whether he/she donated blood in March 2007'],axis=1)
target.value_counts()


# Checking target incidence

# In[17]:


target.value_counts(normalize=True)*100


# Splitting transfusion into train and test datasets
# 

# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data,target,stratify=target,test_size=0.3,random_state=42)
Y_train.value_counts(normalize=True)*100


# In[21]:


Y_test.value_counts(normalize=True)*100


# Selecting model using TPOT

# In[30]:


from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score
model=TPOTClassifier(generations=5,population_size=20,scoring='roc_auc',verbosity=2)
model.fit(X_train,Y_train.values.ravel())
model.export('Bloodtransfusion.py')


# In[34]:


from sklearn.metrics import accuracy_score
Y_pred=model.predict(X_test)
print("\nAUC score:%f"%(roc_auc_score(Y_test,Y_pred)*100))
print("\nAccuracy score:%f"%(accuracy_score(Y_test,Y_pred)*100))


# Checking the variance

# In[35]:


data.var()


#  Log normalization

# In[37]:


import numpy as np
data['LogNormalMonetary']=np.log(data['Monetary (c.c. blood)'])
data=data.drop(['Monetary (c.c. blood)'],axis=1)
data.head()


# In[38]:


data.var()


# Training the logistic regression model

# In[39]:


X_train,X_test,Y_train,Y_test=train_test_split(data,target,test_size=0.3,stratify=target)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train.values.ravel())
Y_pred=lr.predict(X_test)
Y_pred


# In[40]:


print(Y_test.values)


# Conclusion

# In[43]:


from sklearn.metrics import confusion_matrix
print("\nAUC score:%f"%(roc_auc_score(Y_test,Y_pred)*100))
print("\naccuracy score:%f"%(accuracy_score(Y_test,Y_pred)*100))
print(confusion_matrix(Y_test,Y_pred))

