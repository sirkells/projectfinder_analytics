#%%[markdown]
# DOCUMENTATION

#%%[markdown] 
# 1. Import relevant libraries and download relevant resources
# 2. Obtain the details for retrieving and stroing the data
# 3. Get the dataset from mongoDB database and store it as a pandas dataframe. <br/>
# 2. Reduce the orginial dataframe by removing the columns which are not needed for Topic Modelling. Cureently we are considering only the area and the description of the project as the columns in our dataframe. 
# 3. Identifying stopwords:
#   3.1. Load NLTK's English and German stopwords
#   3.2. Add cities and mothns to it 
#   3.3. Manually added stopwords (irrelevant words for our analysis)
# 4. Creation of Stemmer: Creating our own stemmer as a dictionary where we specify how to combine same words
# 5. Tokenzing and Stemming the description data present in the dataframe.
# 6. Get frequency distribution of all words present in the dataframe.
# 7. Choose a threshold frequency for top words(currently 100)
# 8. Reduce the words in tokenized column to these top words for each row
# 9. Remove the rows which have less than 10 tokens
# 10. Create training and test dataset
# 11. Train LDA model
# 12. Save the model as 'LDA_Approach_1.model'





#%%[markdown]
# Importing all the relevant libraries and downloading all relevant resources

#%%

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
import time
from nltk import FreqDist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from nltk.tokenize import word_tokenize
import pickle
import string
from string import punctuation
import os
%matplotlib inline
sns.set_style("darkgrid")

#%%
#Using NLTK Downloader to obtain the resource stopwords, punkt
nltk.download('stopwords')
nltk.download('punkt')

#%%[markdown]
# Database details for retrieving dataset and storing the dataset

#%%
#Details for retrieving  data from projectfinder
db_loc = {
    'ip' :'10.10.250.0',
    'port' : 27017,
    'database' : 'projectfinder',
    'collection' : 'itproject_clean'
}

#%%
#Details for storing data related to projectfinder
db_data = {
    'ip' :'10.10.250.0',
    'port' : 27017,
    'database' : 'projectfinder',
    'collection' : 'mldata1'
}

#%%[markdown]
#Methods for loading the dataset
#%%
def load_dataset_from_mongodb(db_obj):
    
    """
    This method loads a dataset as a pandas dataframe from MongoDB 
    
    Parameters:
    @db_obj (dict): Storing the ip address, port number, database name and collection name for dataset to be loaded
    
    Returns:
    panadas dataframe: Containing the loaded dataset
    """
    
    #Extracting the items from the inputted dictionary
    dbname = db_obj['database']
    ip = db_obj['ip']
    port = db_obj['port']
    collection = db_obj['collection']
    
    #Creating a connection to the database using MongoClient
    connection = MongoClient(ip, port)
    db = connection[dbname]
    
    #Excluding the fileds which are not needed in the dataframe 
    #Currenlty excluding the id associated with each document of the collection
    
    exclude_field = {'_id': False}
    raw_dataset = list(db[collection].find({}, projection=exclude_field))
    
    dataset = pd.DataFrame(raw_dataset)
    print(f'Data loaded from mongodb {collection} collection succesfully')
    return dataset

#%%
def save_to_momgodb(df,db_):
    
    """
    This method saves a dataframe as a collection into a specified MongoDB database.
    
    Parameters:
    @df (pandas dataframe): Storing the dataset to be saved
    @db_ (dict): Details for the database where the given dataset is to be saved
    
    """
    
    #Convert data prsent in the dataframe to JSON format
    data = df.to_dict(orient='records')
    
     #Extracting the items from the inputted dictionary of database details
    dbname = db_['database']
    ip = db_['ip']
    port = db_['port']
    coll = db_['collection']
    
    #Creating a connection to the database using MongoClient
    connection = MongoClient(ip,port)
    db = connection[dbname]
    col = db[collection].insert_many(data)
    
    print(f'data saved as {coll}')

#%%
def load_dataset_from_json(data):
    with open(data) as f:
            d = json.load(f)
        #normalize json
    dataset= json_normalize(d)
    return dataset

#%%
df_rawData = load_dataset_from_mongodb(db_loc)
df_rawData.shape

#%%
def get_required_dataset(original_dataset):
    
    #Select required colunms
    df = original_dataset[['description', 'bereich']]
    df = df[df['description'] != '']
    #df.rename(columns = {'description' : 'project', 'bereich' : 'class'})
    df['project'] = df['description']
    df['label'] = df['bereich']
    df.drop(['description', 'bereich'], axis=1, inplace=True)
    df = df[df['label'] != 'IT/Bauingenieur']
    df = df.drop_duplicates()
    return df

#%%
df_preprocessedDataset = get_required_dataset(df_rawData)
df_preprocessedDataset.shape
df_preprocessedDataset.head()

#%%
# shuffle the data
df_preprocessedDataset = df_preprocessedDataset.sample(frac=1.0)
df_preprocessedDataset.reset_index(drop=True,inplace=True)
df_preprocessedDataset.head()

#%%
df_preprocessedDataset.iloc[0,0]


#%%
# load nltk's German and english stopwords'
currDir = os.getcwd()
print(currDir)
dataDir = os.path.join(currDir,  "ML", "USL", "data")
with open(os.path.join(dataDir, 'german_stopwords_full.txt'), 'r') as f:
    stopwords_germ = f.read().splitlines()
stopwords_eng = nltk.corpus.stopwords.words('english')

#%%
#german cities
from ML.USL.bundeslander import Baden_Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg, Hessen, Mecklenburg_Vorpommern, Niedersachsen, Nordrhein_Westfalen, Rheinland_Pfalz, Saarland, Sachsen, Sachsen_Anhalt, Schleswig_Holstein, Thüringen, Ausland
All = Baden_Württemberg + Bayern + Berlin + Brandenburg + Bremen +Hamburg + Hessen + Mecklenburg_Vorpommern + Niedersachsen + Nordrhein_Westfalen + Rheinland_Pfalz + Saarland + Sachsen + Sachsen_Anhalt + Schleswig_Holstein + Thüringen + Ausland
cities = list(set([city.lower() for city in All]))


#%%
months = ['Januar', 'January','Februar', 'February', 'März', 'March', 'April', 'Mai', 'May', 'Juni', 'June', 'Juli', 
          'July', 'August', 'September', 'Oktober', 'October', 'November', 'Dezember', 'December']
months = [month.lower() for month in months]
print(months)

#%%
stopwords_manual = [line.rstrip('\n') for line in open(os.path.join(dataDir, 'stopwords_manual.txt'))]
print(len(stopwords_manual))

#%%
stopwords_all = list(set(stopwords_germ + stopwords_eng + stopwords_manual + cities + months))
len(stopwords_all)


#%%
stopwords_add = []
stopwords_add = list(set(stopwords_add + stopwords_manual))
checker = list(set(stopwords_germ + stopwords_eng + cities + months))
stopwords_add.sort()
with open('stopwords_manual.txt', 'w') as f:
    for item in stopwords_add:
        if item not in checker:
            f.write("%s\n" % item)


#%%
stopwords_manual = [line.rstrip('\n') for line in open('stopwords_manual.txt')]
print(len(stopwords_manual))

stopwords_all = list(set(stopwords_germ + stopwords_eng + stopwords_manual + cities + months))
len(stopwords_all)

#%%
stemmer_own = {
    
    'abgeschlossen': 'abgeschlossen',
    'admin': 'administration',  
    'verwaltung': 'administration',
    'architektur' : 'architekture',
    'agil' : 'agile',
    'analys': 'analyst',
    'app': 'application',
    'anwend' : 'application',
    'automat': 'automate',
   
    
    'consultant' : 'berater',
    'berat': 'berater',
    'bereich' : 'bereich',
    'cisco': 'cisco',
    'konzept' : 'concept',
    'container': 'containerization',
    'contin': 'continuous',
    'zertifi' : 'certificate',
    'certifi' : 'certificate',
    'design' : 'design',
    'engineer' : 'engineer',
    'ingenieur'  : 'engineer',
    'entwick': 'entwicklung',
    'develop': 'entwicklung',
    'device':'device',
    'program': 'entwicklung',
    'entwickler' : 'entwicklung',
    
    'extern': 'external',
    'framework': 'framework',
    'fix': 'fix',
    'globalen': 'global',
    'install' : 'install',
    'schnittstell': 'interface',
    'implement' : 'implementation', 
    'infrastr' : 'infrastructure',
    'informati' : 'informatik',
    'intern': 'internal',
    'integriert' : 'integrate',
    'konfigur': 'konfigure',
    'manage' : 'management',
    'method' : 'method',
    'überwach' : 'monitoring',
    'mobil': 'mobil',
    'betrieb' : 'operation',
    'künstliche': 'künstliche',
    'notebook': 'notebooks',
    'read':'read',
    'write':'write',
    'relational':'relational',
    'master':'master',
    'script':'script',
    'skript':'skript',
    'skale':'scale',
    
    'operat' : 'operation',
    'operie' : 'operation',
    'vorschläg' : 'option',
    'plattform' : 'platform',
    'projec' : 'project',
    'prozess' : 'process',
    'process' : 'process',
    'bearbeitung' : 'process',
    'scrum': 'scrum',
    'softwar': 'software',
    'spezifi' :'specification',
    'specifi' :'specification',
    'unterstützt' : 'support',
    'support' : 'support',
    'system': 'system',
    'anfoder': 'requirement',
    'tech' : 'tech',
    
}

#%%
def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    cleaned: list of strings. List containing normalized and stemmed word tokens
    """

    try:
        text = re.sub(r'(\d)',' ',text.lower())
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        tokens = word_tokenize(text)
        tokens_cleaned = [word for word in tokens if word not in stopwords_all and len(word) > 1]
        cleaned = []
        stemmer_keys = list(stemmer_own.keys())
        for word in tokens_cleaned:
            for stemmer_key in stemmer_keys:
                if stemmer_key in word:
                    stemmed_word = stemmer_own[stemmer_key]
                    cleaned.append(stemmed_word)
                    break
            else:
                cleaned.append(word)
  
                

    except IndexError:
        pass

    return cleaned

#%%
# Clean text and title and create new column "tokenized"
t1 = time.time()
df_preprocessedDataset['tokenized'] = df_preprocessedDataset['project'].apply(tokenize)
t2 = time.time()
print("Time taken to prepare", len(df), "projects documents:", (t2-t1)/60, "min")

#%%
df_preprocessedDataset.head()

#%%
# Create a list containing all the words in a dataframe
all_words_df = [word for item in list(df['tokenized']) for word in item]

# Use nltk fdist to get a frequency distribution of all words
fdist_words = FreqDist(all_words_df)
print(len(fdist_words)) # number of unique words
print(type(fdist_words))

#print(fdist_words.items())

#%%
total_unique_words = len(fdist_words)
sorted_freqDist_words = fdist_words.most_common()
maxFreq = sorted_freqDist_words[0][1]
print(maxFreq)
freq_values = [sorted_freqDist_words[i][1] for i in range(total_unique_words)]
avgFreq = np.mean(freq_values)
print(avgFreq)

#%%
#Considering words with frequency of 100 or more
top_words = [sorted_freqDist_words[i][0] for i in range(total_unique_words) if sorted_freqDist_words[i][1] >= 100]
print(len(top_words))
#print(top_words)

#%%
def most_appeared(text):
    return [word for word in text if word in top_words]

#%%
#Reduce the words in tokenized column to the words with frequency more than 100. 
df['tokenized'] = df['tokenized'].apply(most_appeared)

#%%
df.head(20)

#%%
# only keep articles with more than 10 tokens, otherwise too short
df = df[df['tokenized'].map(len) >= 10]
# make sure all tokenized items are lists
df = df[df['tokenized'].map(type) == list]
df.reset_index(drop=True,inplace=True)
print("After cleaning and excluding short aticles, the dataframe now has:", len(df), "articles")

#%%
# create a mask of binary values to split into train and test
msk = np.random.rand(len(df)) < 0.9960
msk

#%%
train_df = df[msk]
train_df.reset_index(drop=True,inplace=True)

test_df = df[~msk]
test_df.reset_index(drop=True,inplace=True)

#%%
train_df.head()

#%%
def train_lda(data, n=10):
    """
    This function trains the lda model
    We setup parameters like number of topics, the chunksize to use in Hoffman method
    We also do 2 passes of the data since this is a small dataset, so we want the distributions to stabilize
    """
    num_topics = n
    chunksize = 300
    dictionary = corpora.Dictionary(data['tokenized'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
    t1 = time.time()
    # low alpha means each document is only represented by a small number of topics, and vice versa
    # low eta means each topic is only represented by a small number of words, and vice versa
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)
    t2 = time.time()
    print("Time to train LDA model on ", len(df), "documents: ", (t2-t1)/60, "min")
    return dictionary,corpus,lda

#%%
dictionary,corpus,lda = train_lda(train_df, 10)

#%%
lda.save('LDA_Approach_1.model')

#%%%
from gensim import corpora, models, similarities
# later on, load trained model from file
model =  models.LdaModel.load('LDA_Approach_1.model')

# print all topics
model.show_topics(num_topics=20, num_words=20)

#%%import pickle
with open('dictionary', 'wb') as output:
    pickle.dump(dictionary, output)
    
with open('corpus', 'wb') as output:
    pickle.dump(corpus, output)