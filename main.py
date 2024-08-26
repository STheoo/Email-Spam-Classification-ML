#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string




df = pd.read_csv("train.csv")
df['email'] = df['email'].str.lower()
df.rename(columns={'label': 'spam'}, inplace=True)
df['spam'] = df['spam'].apply(lambda x: 1 if x == 'spam' else 0)
df.head(5)


# In[2]:


df.duplicated().sum()


# In[3]:


df.drop_duplicates(inplace=True)


# In[4]:


df.isnull().sum()


# In[5]:


df.shape


# In[6]:


nltk.download('stopwords')


# In[7]:


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words()]
    
    #PorterStemmer seemed to worsen results
    
    return clean_words


# In[8]:


df['email'].head().apply(process_text)


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer

email_bow = CountVectorizer(analyzer = process_text).fit_transform(df['email'])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(email_bow, df['spam'], test_size = 0.20, random_state = 0)


# In[ ]:


email_bow.shape


# In[ ]:


from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


svc = SVC(kernel='rbf', gamma='scale', class_weight='balanced', C = 3)
knc = KNeighborsClassifier(n_neighbors=8, weights='uniform', p=2)
mnb = MultinomialNB(alpha = 1.1)
dtc = DecisionTreeClassifier()
bc = BaggingClassifier(n_estimators=70, n_jobs=-1)
gbdt = GradientBoostingClassifier(n_estimators=250,random_state=2)


# In[ ]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'BgC': bc, 
    'GBDT':gbdt,
}


# In[ ]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[ ]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[ ]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
performance_df


# In[ ]:


gbdt.fit(X_train,y_train)
y_pred = gbdt.predict(X_test)
confusion_matrix(y_test,y_pred)


# In[ ]:


from sklearn.model_selection import cross_val_score
cv = cross_val_score(gbdt, X_train, y_train, cv=5)


# In[ ]:


cv.mean()


# In[45]:


def train_test(train_file, test_file):
    
    train_data = pd.read_csv(train_file)
    train_data['email'] = train_data['email'].str.lower()
    train_data.rename(columns={'label': 'spam'}, inplace=True)
    train_data['spam'] = train_data['spam'].apply(lambda x: 1 if x == 'spam' else 0)
    df.drop_duplicates(inplace=True)
    
    X_train = train_data['email']
    y_train = train_data['spam']
    X_train = X_train.drop(X_train.index[10:])
    y_train = y_train.drop(y_train.index[10:])


    X_train = CountVectorizer(analyzer = process_text).fit_transform(X_train)


    model = GradientBoostingClassifier(n_estimators=250,random_state=2)
    model.fit(X_train, y_train)


    test_data = pd.read_csv(test_file)

    X_test = test_data['email']

    X_test = CountVectorizer(analyzer = process_text).fit_transform(X_train)

    y_pred = model.predict(X_test)

    output_file = 'predictions.txt'
    with open(output_file, 'w') as file:
        for prediction in y_pred:
            if prediction is 1:
                file.write("spam" + '\n')
            else
                file.write("ham" + '\n')


# In[ ]:


train_test("train.csv","test.csv")


# In[ ]:




