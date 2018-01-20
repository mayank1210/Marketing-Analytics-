
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[73]:


pulse = pd.read_csv('Pulse.csv')[1:]


# In[75]:


pulse.head()


# In[77]:


pulse.info()


# In[78]:


pulse.describe()


# In[79]:


pulse.columns


# In[80]:


sns.pairplot(pulse)


# In[81]:


sns.distplot(pulse['Advertising'])


# In[82]:


sns.heatmap(pulse.corr())


# In[83]:


X = pulse.drop('Advertising', axis=1)
y = pulse['Advertising']


# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=101)


# In[86]:


from sklearn.linear_model import LinearRegression


# In[87]:


lm = LinearRegression()


# In[88]:


lm.fit(X_train,y_train)


# In[89]:


print(lm.intercept_)


# In[90]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[91]:


predictions = lm.predict(X_test)


# In[92]:


plt.scatter(y_test,predictions)


# In[93]:


sns.distplot((y_test-predictions),bins=50);


# In[94]:


from sklearn import metrics


# In[95]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

