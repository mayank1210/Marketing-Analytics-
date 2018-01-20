
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[35]:


teens = pd.read_csv('snsdata.csv')


# In[36]:


teens.info()


# In[37]:


teens.head()


# In[38]:


teens.describe()


# In[39]:


teens['age'].describe()


# In[40]:


# Managing Outliers

def impute_age(cols):
    age = cols[0]
    if age >= 20:
        age = None
    else: 
        if age < 13:
            age = None
        else:
            return age;


# In[41]:


teens['age'] = teens[['age']].apply(impute_age, axis = 1)
teens['age'].describe()


# In[42]:


teens['gender'].value_counts(dropna = False)


# In[43]:


sns.countplot(x= 'gender', data=teens)


# In[44]:


dummies= pd.get_dummies(data = teens['gender'])
dummies.head()


# In[45]:


teens= pd.concat([teens, dummies], axis= 1)
teens.info()


# In[46]:


teens[['gender', 'F', 'M']].head()


# In[47]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='gradyear',y='age',data=teens,palette='winter')


# In[48]:


GradyearMeansByAge = teens['age'].groupby(teens['gradyear']).mean()


# In[49]:


print(GradyearMeansByAge)


# In[50]:


def impute_age(cols):
    Age = cols[0]
    Gradyear = cols[1]
    
    if pd.isnull(Age):

        if Gradyear == 2006:
            return 18.655858

        elif Gradyear == 2007:
            return 17.706172
        
        elif Gradyear == 2008:
            return 16.767701

        else:
            return 15.819573

    else:
        return Age


# In[51]:


teens['age'] = teens[['age','gradyear']].apply(impute_age,axis=1)


# In[52]:


teens['age'].describe()


# In[53]:


interests= teens.iloc[:,4:40]
interests.info()


# In[54]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()


# In[55]:


scaler.fit(interests)


# In[56]:


scaled_features= scaler.transform(interests)
interests_z = pd.DataFrame(scaled_features, columns= interests.columns)
interests_z.describe()


# In[57]:


from sklearn.cluster import KMeans


# In[58]:


kmeans = KMeans(n_clusters=5)


# In[59]:


kmeans.fit(interests_z)


# In[60]:


kmeans.labels_


# In[61]:


labels = pd.DataFrame(kmeans.labels_)


# In[62]:


teens_labels = pd.concat([teens,labels], axis=1)


# In[63]:


teens_labels.rename(columns={0: 'labels'}, inplace=True)


# In[64]:


teens_labels.head()


# In[65]:


teens_labels['labels'].value_counts()


# In[66]:


AgeMeansByLabels = teens_labels['age'].groupby(teens_labels['labels']).mean()
print(AgeMeansByLabels)


# In[67]:


FemaleMeansByLabels = teens_labels['F'].groupby(teens_labels['labels']).mean()
print(FemaleMeansByLabels)


# In[68]:


FriendsMeanByLabels = teens_labels['friends'].groupby(teens_labels['labels']).mean()
print(FriendsMeanByLabels)


# In[69]:


teens.columns


# In[77]:


from scipy.spatial.distance import cdist
distortions = []
for i in range(1,40):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(interests_z)
    distortions.append(sum(np.min(cdist(interests_z, kmeans.cluster_centers_, 'euclidean'), axis=1)) / interests_z.shape[0])


# In[80]:


plt.plot(range(1,40), distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

