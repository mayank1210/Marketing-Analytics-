
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
get_ipython().magic(u'matplotlib inline')


# In[2]:


messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()


# In[3]:


messages.info()


# In[4]:


messages.groupby('label').describe()


# In[5]:


messages['length'] = messages['message'].apply(len)
messages.head()


# In[6]:


messages.length.describe()


# In[7]:


messages.hist(column='length', by='label', bins=50,figsize=(12,4))


# In[16]:


nltk.download('stopwords')
from nltk.corpus import stopwords
import string
stopwords.words('english')


# In[17]:


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[18]:


messages['message'].apply(text_process)


# In[20]:


messages.head()


# In[21]:


from wordcloud import WordCloud


# In[22]:


ham_words = ' '.join(list(messages[messages['label']=='ham']['message']))
spam_words = ' '.join(list(messages[messages['label']=='spam']['message']))


# In[23]:


spam_wordcloud= WordCloud(width=800, height=600, background_color='white', max_words=50,colormap='magma').generate(spam_words)
plt.figure( figsize=(10,8))
plt.imshow(spam_wordcloud)


# In[24]:


ham_wordcloud= WordCloud(width=800, height=600, background_color='white', max_words=50,colormap='magma').generate(ham_words)
plt.figure( figsize=(10,8))
plt.imshow(ham_wordcloud)


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer


# In[26]:


bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# In[27]:


messages_bow = bow_transformer.transform(messages['message'])


# In[28]:


print('Shape of the Sparse Matrix: ', messages_bow.shape)


# In[29]:


messages_bow.nnz


# In[30]:


sparsity = (100*messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))


# In[31]:


print('sparsity:', sparsity)


# In[32]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[33]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[34]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[35]:


from sklearn.model_selection import train_test_split
X= messages_tfidf
y= messages['label']


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[54]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# In[55]:


def train_classifier(clf, X_train, y_train):    
    clf.fit(X_train, y_train)


def predict_labels(clf, features):
    return (clf.predict(features))


# In[56]:


A = MultinomialNB()
B = DecisionTreeClassifier()
C = AdaBoostClassifier()
D = KNeighborsClassifier()
E = RandomForestClassifier()
F= LogisticRegression()
G= SVC()
H= MLPClassifier()


# In[59]:


clf = [A,B,C,D,E,F,G,H]
pred_val = [0,0,0,0,0,0,0,0]

for a in range(0,8):
    train_classifier(clf[a], X_train, y_train)
    y_pred = predict_labels(clf[a],X_test)
    pred_val[a] = f1_score(y_test, y_pred, average= "binary", pos_label= 'spam') 
    print (pred_val[a])


# In[60]:


models = ('Multi-NB', 'DTs', 'AdaBoost', 'KNN', 'RF','LR', 'SVM', 'Neural')
y_pos = np.arange(len(models))
y_val = [ x for x in pred_val]
plt.bar(y_pos,y_val,)
plt.xticks(y_pos, models)
plt.ylabel('Accuracy Score')
plt.title('Accuracy of Models')


# In[61]:


spam_detect_model = MultinomialNB()


# In[62]:


spam_detect_model.fit(X_train, y_train)


# In[63]:


predictions = spam_detect_model.predict(X_test)


# In[64]:


from sklearn.metrics import classification_report


# In[65]:


print (classification_report(y_test, predictions))


# In[66]:


from sklearn.model_selection import GridSearchCV


# In[67]:


param_grid = {'alpha': [0, 1, 2, 5,10]} 


# In[68]:


grid = GridSearchCV( MultinomialNB(),param_grid,refit=True,verbose=3)


# In[69]:


grid.fit(X_train,y_train)


# In[70]:


grid.best_params_


# In[71]:


grid_predictions = grid.predict(X_test)


# In[72]:


print (classification_report(y_test, grid_predictions))

