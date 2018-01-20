
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[18]:


df = pd.DataFrame()

with open('groceries.csv', 'r') as f:
    for line in f:
        df = pd.concat( [df, pd.DataFrame([tuple(line.strip().split(','))])], ignore_index=True )


# In[8]:


df.info()


# In[9]:


df.head()


# In[19]:


numpyMatrix = df.as_matrix()


# In[21]:


numpyMatrix


# In[22]:


from mlxtend.preprocessing import OnehotTransactions


# In[23]:


oht = OnehotTransactions()
oht_ary = oht.fit(numpyMatrix).transform(numpyMatrix)
dataframe= pd.DataFrame(oht_ary, columns=oht.columns_)


# In[35]:


dataframe.drop(dataframe.columns[0], axis=1)


# In[56]:


from mlxtend.frequent_patterns import apriori

frequent_itemsets= apriori(dataframe, min_support=0.05, use_colnames=True)
frequent_itemsets


# In[57]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules

