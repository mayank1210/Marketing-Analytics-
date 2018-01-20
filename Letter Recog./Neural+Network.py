
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[14]:


concrete= pd.read_csv('Concrete_Data.csv',names=['cement','slag','ash','water','superplastic','coarseagg','fineagg', 'age','strength'])


# In[15]:


concrete.head()


# In[16]:


concrete.info()


# In[17]:


concrete.describe()


# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


min_max_scaler= MinMaxScaler()


# In[23]:


concrete = pd.DataFrame(min_max_scaler.fit_transform(concrete), columns=['cement','slag','ash','water','superplastic','coarseagg','fineagg', 'age','strength'] )


# In[24]:


concrete.describe()


# In[25]:


from sklearn.model_selection import train_test_split


# In[27]:


X= concrete.drop('strength', axis=1)


# In[28]:


y= concrete['strength']


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[30]:


from sklearn.neural_network import MLPRegressor


# In[33]:


model= MLPRegressor(hidden_layer_sizes=(1,))


# In[34]:


model.fit( X_train, y_train)


# In[35]:


model.n_layers_


# In[43]:


predictions= model.predict(X_test)
print(predictions)


# In[60]:


plt.figure(figsize=(12,8))
plt.scatter(predictions, y_test)


# In[62]:


np.corrcoef(predictions, y_test)


# In[67]:


model= MLPRegressor(hidden_layer_sizes=(10,))


# In[68]:


model.fit(X_train, y_train)


# In[69]:


predictions2 = model.predict(X_test) 


# In[70]:


np.corrcoef(predictions2, y_test)


# In[71]:


plt.figure(figsize=(12,8))
plt.scatter(predictions2, y_test)


# In[72]:


model= MLPRegressor()


# In[73]:


model.fit(X_train, y_train)


# In[74]:


predictions3 = model.predict(X_test)


# In[77]:


plt.figure(figsize=(12,8))
plt.scatter(predictions3, y_test)


# In[79]:


np.corrcoef(predictions3, y_test)

