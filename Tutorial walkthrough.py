#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

# Load CSV master file
csv_file = 'P:\\Desktop\\Thesis data\\master.csv'

# The first row in the CSV file contains the headers
df = pd.read_csv(csv_file)

# Replace NaN values in the 'Text files' column with an empty string
df['Text files'].fillna('', inplace=True)

X = df['Text files']

y = np.asarray(df[df.columns[1:29]])

print(len(y))

# Transform the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=30, max_df=0.9)
X_tfidf = vectorizer.fit_transform(X)

print(X_tfidf)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=101)

# Build a multi-output classifier with Logistic Regression as the base estimator
clf = MultiOutputClassifier(LogisticRegression()).fit(X_train, y_train)

# Now you can use 'clf' to make predictions on your multi-label classification task
# For example:
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy Score: ', accuracy_score(y_test, y_pred))


# In[3]:


# calculate metrics
from sklearn.metrics import classification_report
periods= ['early medieval', 'late mesolithic', 'medieval', 'post medieval',
    'later prehistoric', 'early iron age', 'middle palaeolithic', 'neolithic',
    'late iron age', 'bronze age', 'early bronze age', 'late prehistoric', 'roman',
    'middle iron age', 'late neolithic', 'early neolithic', 'middle bronze age',
    'early mesolithic', 'lower palaeolithic', 'upper palaeolithic', 'late bronze age',
    'palaeolithic', 'early prehistoric', 'mesolithic', '20th century',
    'middle neolithic', 'iron age', 'nil antiquity']
print(classification_report(y_test, y_pred, target_names=periods))


# In[4]:


# Classifiers
print(len(y_train))
from sklearn.svm import SVC
clf = MultiOutputClassifier(SVC()).fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
print(classification_report(y_test, y_pred_svc, target_names=periods))


# In[5]:


from sklearn.ensemble import RandomForestClassifier
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=10)).fit(X_train, y_train)

y_pred_rf = clf.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=periods))


# In[6]:


from sklearn.tree import DecisionTreeClassifier
clf = MultiOutputClassifier(DecisionTreeClassifier()).fit(X_train, y_train)
y_pred_dt = clf.predict(X_test)
print(classification_report(y_test, y_pred_dt, target_names=periods))


# In[ ]:




