
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[96]:


continous= pd.read_csv('continous.csv')[1:]


# In[97]:


continous.head()


# In[99]:


continous.info()


# In[100]:


continous.describe()


# In[101]:


continous.columns


# In[102]:


sns.pairplot(continous)


# In[103]:


sns.distplot(continous['Advertising'])


# In[104]:


sns.heatmap(continous.corr())


# In[105]:


X = continous.drop('Advertising', axis=1)
y = continous['Advertising']


# In[106]:


from sklearn.model_selection import train_test_split


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=101)


# In[108]:


from sklearn.linear_model import LinearRegression


# In[109]:


lm = LinearRegression()


# In[110]:


lm.fit(X_train,y_train)


# In[111]:


print(lm.intercept_)


# In[112]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[113]:


predictions = lm.predict(X_test)


# In[114]:


plt.scatter(y_test,predictions)


# In[115]:


sns.distplot((y_test-predictions),bins=50);


# In[116]:


from sklearn import metrics


# In[117]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

