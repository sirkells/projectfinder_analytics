#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json, nltk, re, os, codecs

import pandas as pd
from pandas.io.json import json_normalize
#import pandas_profiling
import numpy as np

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style("whitegrid")
import altair as alt
alt.renderers.enable("notebook")

# Code for hiding seaborn warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn import feature_extraction

from pymongo import MongoClient


# In[2]:



# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
import warnings

warnings.simplefilter('ignore')

import pickle
from sklearn.externals import joblib


# In[3]:


def load_data_from_momgodb(db_obj):
    dbname = db_obj['db']
    ip = db_obj['ip']
    port = db_obj['port']
    coll = db_obj['coll']
    connection = MongoClient(ip,port)
    db = connection[dbname]
    exclude_data = {'_id': False}
    raw_data = list(db[coll].find({}, projection=exclude_data))
    dataset = pd.DataFrame(raw_data)
    print(f'Data loaded from mongodb {coll} collection succesfully')
    return dataset


# In[4]:


db_data = {
    'ip' :'10.10.250.0',
    'port' : 27017,
    'db' : 'projectfinder',
    'coll' : 'mldata'
}


# In[23]:


ready_data = load_data_from_momgodb(db_data)
ready_data.head()


# In[ ]:


categories = ready_data.categories.str.split(pat=';',expand=True)


# In[12]:


#load saved model
import pickle
from sklearn.externals import joblib
filename = '../stopwords.sav'
stopwords = joblib.load(filename)


# In[13]:


def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Stem word tokens and remove stop words
    stemmer_eng = SnowballStemmer("english", ignore_stopwords=True)
    stemmer_germ = SnowballStemmer("german", ignore_stopwords=True) 
    try:
        # Convert text to lowercase and remove punctuation
        text = re.sub("[^a-zA-Z ]", " ", text.lower()) #remove non alphbetic text

        # Tokenize words
        tokens = word_tokenize(text)
        stemmed = [stemmer_germ.stem(word) for word in tokens if word not in stopwords]
        stemmed = [stemmer_eng.stem(word) for word in stemmed if len(word) > 1]
        stemmed = [word for word in stemmed if word not in stopwords]
    except IndexError:
        pass

    return stemmed


# In[14]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# In[15]:


tokenize('entwickler entwicklung bin ich i love running i am a jumped')


# In[16]:


feature = ready_data['project']
target = ready_data.drop('project', axis=1)
target_name = list(target.columns.values)


# In[18]:


#filename = 'models/feature.sav'
joblib.dump(feature, '../models/feature.sav')
joblib.dump(target, '../models/target.sav')


# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(feature, target, random_state = 1)

np.random.seed(17)
pipeline.fit(X_train, Y_train)


# In[18]:


#save model

filename = 'model.pkl'
joblib.dump(pipeline, filename)


# In[ ]:





# In[13]:


def get_eval_metrics(actual, predicted, col_names):
    """Calculate evaluation metrics for ML model
    
    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    col_names: list of strings. List containing names for each of the predicted fields.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i])
        recall = recall_score(actual[:, i], predicted[:, i])
        f1 = f1_score(actual[:, i], predicted[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df


# In[14]:


# Calculate evaluation metrics for training set
Y_train_pred = pipeline.predict(X_train)


# In[16]:


col_names = list(target.columns.values)

print(get_eval_metrics(np.array(Y_train), Y_train_pred, col_names))


# In[19]:


# Calculate evaluation metrics for test set
Y_test_pred = pipeline.predict(X_test)

eval_metrics0 = get_eval_metrics(np.array(Y_test), Y_test_pred, col_names)
print(eval_metrics0)


# In[7]:


#load saved model
loaded_pipeline = joblib.load('../models/model.pkl')


# In[19]:


#load feature and target data
loaded_feature = joblib.load('../models/feature.sav')
loaded_target = joblib.load('../models/target.sav')


# In[ ]:


#Y_pred = loaded_pipeline.predict(X_train)


# In[20]:


def get_category(text, model, labels):
    class_labels = model.predict([text])[0]
    classification_results = dict(zip(labels.columns, class_labels))
    return classification_results
    


# In[21]:


text_data ="""Für unseren Kunden in Nordrhein-Westfalen suchen wir ab sofort eine/n Senior Berater SAP TM (m/w/d).


 


Ihre Aufgaben:


- Erarbeitung und Umsetzung von neuen Konzepten im Rahmen unseres internationalen Rolloutprojektes


- Customizing/Konfiguration in den Bereichen SAP TM


- Erstellung von Spezifikationen für Entwicklungen und Schnittstellen


- Vorbereitung und Durchführung von Tests und User Trainings, sowie Unterstützung im Rahmen der Hypercare


 


Ihre Qualifikation:


- Sehr gute Kenntnisse von SAP TM und der Integration in die angrenzende Module (SD, MM, FI/CO)


- Erfahrung in der Implementierung von SAP TM für Verlader mit LKW-, Schiffscontainer-, Luftfracht- und Bahn-Transporten (Outbound & Inbound)


- Vorzugsweise Erfahrungen im Rahmen der Gefahrguttransporte


- Expertise in komplexen Frachtkostenabwicklungen


- Schnittstelle / Kommunikation über EDI


- Erfahrungen in der Integration zu SAP GTS und LBN


""" #ready_data['project'][0]
#text_data = tokenize(text_data)
#text_data = ' '.join(text_data)
text_data


# In[22]:


get_category(text_data, loaded_pipeline, loaded_target)


# In[49]:


class_labels = pipeline.predict([text_data])[0]


# In[50]:


classification_results = dict(zip(target.columns, class_labels))


# In[51]:


classification_results


# In[52]:


class_labels


# In[ ]:




