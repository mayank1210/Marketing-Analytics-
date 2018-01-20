
# coding: utf-8

# In[250]:


import pandas as pd
import numpy as np
import xlrd
import random 
import math
import matplotlib.pyplot as plt
import seaborn as sns


# In[251]:


get_ipython().magic('matplotlib inline')


# In[252]:


df = pd.read_csv('credit.csv')
df.head()


# In[253]:


df.info()


# In[254]:


df.describe()


# In[255]:


#converting string to integer
#default means that whether the loan applicant was unable to meet the agreed payment terms and went into default. 
#A total of 30 percent of the loans went into default that means 1 is bad and 0 is good number. 
#Bank wants to increase 0 and not 1.

df['default'].replace('yes', 1,inplace=True)
df['default'].replace('no', 0,inplace=True)
df['phone'].replace('yes', 1,inplace=True)
df['phone'].replace('no', 0,inplace=True)


# In[256]:


df['default'].value_counts()


# In[257]:


df.nunique()


# In[258]:


# Initial Data Exploration


# In[259]:


plt.figure(figsize=(10,6))
sns.lmplot('months_loan_duration','amount', df)


# In[260]:


plt.figure(figsize=(11,7))
df[df['default']==1]['amount'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Default=1')
df[df['default']==0]['amount'].hist(alpha=0.5,color='red',
                                              bins=30,label='Default=0')
plt.legend()
plt.xlabel('Amount')
plt.ylabel('Count')
#this shows that most of the people who borrow less are the defaulters.


# In[261]:


plt.figure(figsize=(11,7))
sns.countplot(x='credit_history',hue='default',data=df,palette='Set1')
#this gives the better undersyamding of where are the maximum defaulters
#most of the defaulters come from the "good" and "critical" credit history.


# In[262]:


sns.boxplot(x='default',y='amount',data=df,palette='winter')


# In[263]:


sns.countplot(x='checking_balance',hue='default',data=df,palette='Set1')


# In[264]:


sns.countplot(x='job',hue='default',data=df,palette='Set1')
sns.pairplot


# In[265]:


df['years_at_residence'].value_counts(dropna = False)


# In[266]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap = 'coolwarm', annot=True)


# In[267]:


cat_cb1 = ['savings_balance','checking_balance','months_loan_duration','credit_history','purpose','employment_duration',
          'other_credit','housing','job']


# In[268]:


df2 = pd.get_dummies(df,columns=cat_cb1,drop_first=True)


# In[269]:


df2.head()


# In[270]:


df2.info()


# In[271]:


from sklearn.model_selection import train_test_split


# In[272]:


X1 = df2.drop(['default'],axis=1)
y1 = df2['default']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.30, random_state=101)


# In[273]:


X1.info()


# In[274]:


from sklearn.tree import DecisionTreeClassifier


# In[275]:


dtree1 = DecisionTreeClassifier()


# In[276]:


dtree1.fit(X1_train,y1_train)


# In[277]:


predictions1 = dtree1.predict(X1_test)


# In[278]:


from sklearn.metrics import classification_report,confusion_matrix


# In[279]:


print(classification_report(y1_test,predictions1))
print('/')
print(confusion_matrix(y1_test,predictions1))


# In[280]:


from sklearn.ensemble import RandomForestClassifier


# In[281]:


rfc1 = RandomForestClassifier(n_estimators=6)


# In[282]:


rfc1.fit(X1_train,y1_train)


# In[283]:


rfc1_predictions1 = rfc1.predict(X1_test)


# In[284]:


print(classification_report(y1_test,rfc1_predictions1))
print('/')
print(confusion_matrix(y1_test,rfc1_predictions1))


# In[285]:


rfc1.n_estimators


# In[286]:


from sklearn.grid_search import GridSearchCV


# In[287]:


from sklearn.metrics import roc_auc_score


# In[288]:


param_grid = {'n_estimators': [10, 50, 100, 150, 200, 300, 400]}


# In[289]:


grid1 = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose = 3)
grid1.fit(X1_train, y1_train)


# In[290]:


grid1.best_estimator_


# In[291]:


grid1.best_params_


# In[292]:


grid_rfc1_prediction = grid1.predict(X1_test)
print(classification_report(y1_test,grid_rfc1_prediction))
print('/')
print(confusion_matrix(y1_test,grid_rfc1_prediction))


# In[293]:


# We can clearly see that the F1-score got increased from decision tree to Random forest and then Grid Search Cross Validaiton
# Decision Tree = 0.66
# Random Forest = 0.67
# Grid Search CV = 0.69

