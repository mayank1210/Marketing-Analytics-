
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[4]:


letters = pd.read_csv('letter-recognition.csv', names= ['lettr','x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx'])


# In[5]:


letters.head()


# In[6]:


letters.info()


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X= letters.drop('lettr', axis=1)


# In[10]:


y= letters['lettr']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


# In[17]:


from sklearn.svm import SVC


# In[26]:


model= SVC(kernel='linear')


# In[27]:


model.fit(X_train, y_train)


# In[28]:


predictions= model.predict(X_test)


# In[29]:


print(predictions)


# In[30]:


from sklearn.metrics import classification_report, confusion_matrix


# In[31]:


print(classification_report(y_test, predictions))


# In[25]:


print(confusion_matrix(y_test, predictions))


# In[32]:


model= SVC()


# In[33]:


model.fit(X_train, y_train)


# In[34]:


predictions2 = model.predict(X_test)


# In[35]:


print(predictions2)


# In[37]:


print(classification_report(y_test, predictions2))


# In[38]:


print(confusion_matrix(y_test, predictions2))

