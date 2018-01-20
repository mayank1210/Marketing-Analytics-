
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[23]:


diet = pd.read_csv('diet.csv')[1:]


# In[24]:


diet.head()


# In[25]:


diet.info()


# In[26]:


diet.describe()


# In[27]:


diet.columns


# In[28]:


sns.pairplot(diet.dropna())


# In[29]:


sns.distplot(diet['Sales'])


# In[30]:


sns.heatmap(diet.corr())


# In[31]:


X = diet.drop('Sales', axis=1)
y = diet['Sales']


# In[32]:


from sklearn.model_selection import train_test_split


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=101)


# In[41]:


from sklearn.linear_model import LinearRegression


# In[42]:


lm = LinearRegression()


# In[43]:


lm.fit(X_train,y_train)


# In[44]:


print(lm.intercept_)


# In[45]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[46]:


predictions = lm.predict(X_test)


# In[47]:


plt.scatter(y_test,predictions)


# In[48]:


sns.distplot((y_test-predictions),bins=50);


# In[49]:


from sklearn import metrics


# In[50]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

