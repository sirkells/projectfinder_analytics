#%% 
from hdfs import InsecureClient
import pandas as pd
#%% 
hdfs_client = InsecureClient('http://10.10.250.10:50070', timeout=1)
hdfs_client
#%% 
from datetime import date

# hdfs_path = '/projects/projectfinder/raw/items/' +\
#    date.today().year.__str__() + '/' +\
#    date.today().month.__str__() + '/'

#%% 
hdfs_path = '/projects/projectfinder/raw/items/2019'

#%%
hdfs_client.download(hdfs_path, 'hdfs_data', n_threads=5)


#%% 

hdfs_client_status = hdfs_client.status('/', strict=True)
hdfs_client_status

#%% 
hdfs_file_status = hdfs_client.list(hdfs_path)
hdfs_file_status


#%% [markdown]
# Go to [manuel](https://hdfscli.readthedocs.io/en/latest/advanced.html#path-expansion)
# ```bash
# # install hdfs using pip
# pip install hdfs
# ```

#%%
fnames = hdfs_client.list(hdfs_path)
fnames


#%%
with hdfs_client.read(hdfs_path, encoding = 'utf-8') as reader:
    df = reader
    df

#%%
# Creating a simple Pandas DataFrame
liste_hello = ['hello1','hello2']
liste_world = ['world1','world2']
df = pd.DataFrame(data = {'hello' : liste_hello, 'world': liste_world})
df

#%%# Writing Dataframe to hdfs
with hdfs_client.write('helloworld.csv', encoding = 'utf-8') as writer:
    df.to_csv(writer)


#%%
import os, shutil

directory = 'hdfs_data/2019/'
#os.mkdir('new')
index = 1
for file in sorted(os.listdir(directory)):
      #shutil.copy(file[0], 'new')
      new_dir = directory + file + '/'
      dir_list = os.listdir(new_dir)
      print(index)
      shutil.copy2(f'hdfs_data/2019/{index}/{dir_list[0]}', 'new')
      index = index + 1
      #print(os.listdir(new_dir[0]))


#%%

import json, nltk, re, os, codecs

import pandas as pd
from pandas.io.json import json_normalize
#import pandas_profiling
import numpy as np

from matplotlib import pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_style('darkgrid')

from sklearn import feature_extraction

from pymongo import MongoClient


def clean(data):
    with open(data) as f:
        d = json.load(f)
    #normalize json
    data_norm = json_normalize(d)
    #select colunms
    data_norm = data_norm[['title', 'description', 'bereich.group', 'bereich.group_type']]
    data_norm = data_norm[data_norm['bereich.group'] != 'Others']
    data_norm = data_norm[data_norm['description'] != '']
    data_norm['label'] = data_norm['bereich.group'] + '/' + data_norm['bereich.group_type']
    data_norm.drop(['bereich.group', 'bereich.group_type'], axis=1, inplace=True)
    data_norm['label'].replace('Data Science/Machine Learning', 'DS/Big Data', inplace=True)
    data_norm['label'].replace('Data Science/Business Intelligence', 'DS/BI', inplace=True)
    data_norm['label'].replace('Data Science/Big Data', 'DS/Big Data', inplace=True)
    data_norm['label'].replace('Development/Web', 'Dev/Web', inplace=True)
    data_norm['label'].replace('Development/Mobile', 'Dev/Mob', inplace=True)
    data_norm['label'].replace('Infrastructure/ERP', 'IT/ERP', inplace=True)
    data_norm['label'].replace('Infrastructure/Admin', 'IT/Admin', inplace=True)
    data_norm['label'].replace('Infrastructure/Admin', 'IT/Admin', inplace=True)
    
    return data_norm