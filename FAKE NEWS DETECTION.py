#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np


# In[37]:


import itertools


# In[38]:


df=pd.read_csv("news.csv")


# In[39]:


df.head()


# In[40]:


df.shape


# In[41]:


df.isnull().sum()


# In[42]:


labels=df.label


# In[43]:


labels


# In[44]:


from sklearn.model_selection import train_test_split 


# In[45]:


x_train,x_test,y_train,y_test=train_test_split(df["text"],labels,test_size=0.2,random_state=20)


# In[46]:


x_train.head()


# In[47]:


y_train.head()


# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


# In[49]:


vector=TfidfVectorizer(stop_words='english',max_df=0.7)


# In[50]:


tf_train=vector.fit_transform(x_train)
tf_test=vector.transform(x_test)


# In[51]:


pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tf_train,y_train)


# In[52]:


from sklearn.metrics import accuracy_score,confusion_matrix
y_pred=pac.predict(tf_test)


# In[56]:


score=accuracy_score(y_test,y_pred)


# In[58]:


print(f"Accuracy:{round(score*100,2)}%")


# In[60]:


confusion_matrix(y_test,y_pred,labels=["FAKE","REAL"])


# In[ ]:





# In[ ]:




