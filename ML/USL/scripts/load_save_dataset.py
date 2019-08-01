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
    
def load_dataset_from_json(data):
    with open(data) as f:
        d = json.load(f)
        #normalize json
    dataset= json_normalize(d)
    return dataset