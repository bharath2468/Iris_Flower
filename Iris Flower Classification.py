#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import seaborn as sns


# In[98]:


df= pd.read_csv("C:/Users/Bharath/Downloads/Iris.csv")


# In[99]:


df.head()


# In[100]:


df.isnull().sum()


# In[101]:


df=df.drop(columns="Id")


# In[102]:


df['Species'].value_counts()



# In[103]:


x=df.iloc[:,:4]
y=df.iloc[:,4]


# In[104]:


x


# In[105]:


from sklearn.model_selection import train_test_split


# In[106]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[107]:


from sklearn.linear_model import LogisticRegression


# In[108]:


model=LogisticRegression()


# In[109]:


model.fit(x_train,y_train)


# In[110]:


y_pred=model.predict(x_test)


# In[111]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)


# In[112]:


accuracy=accuracy_score(y_test,y_pred)*100


# In[113]:


print("Accuracy of Iris Flower Classification is {:.2f}".format(accuracy))

