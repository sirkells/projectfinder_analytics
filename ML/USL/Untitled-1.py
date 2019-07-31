#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ML/USL'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Importing Libraries

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
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
sns.set_style("darkgrid")
from pymongo import MongoClient
from nltk.tokenize import word_tokenize
import pickle
import string
from string import punctuation


#%%
#Details for getting data from projectfinder
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


#%%
def load_dataset_from_momgodb(db_obj):
    
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
df = load_dataset_from_momgodb(db_loc)
df.shape


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
df = get_required_dataset(df)
df.shape
df.head()


#%%
# shuffle the data
df = df.sample(frac=1.0)
df.reset_index(drop=True,inplace=True)
df.head()


#%%
df.iloc[0,0]


#%%
# load nltk's German and english stopwords'
import nltk
with open('../german_stopwords_full.txt', 'r') as f:
    stopwords_germ = f.read().splitlines()
stopwords_eng = nltk.corpus.stopwords.words('english')


#%%
#german cities
from bundeslander import Baden_Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg, Hessen, Mecklenburg_Vorpommern, Niedersachsen, Nordrhein_Westfalen, Rheinland_Pfalz, Saarland, Sachsen, Sachsen_Anhalt, Schleswig_Holstein, Thüringen, Ausland
All = Baden_Württemberg + Bayern + Berlin + Brandenburg + Bremen +Hamburg + Hessen + Mecklenburg_Vorpommern + Niedersachsen + Nordrhein_Westfalen + Rheinland_Pfalz + Saarland + Sachsen + Sachsen_Anhalt + Schleswig_Holstein + Thüringen + Ausland
cities = list(set([city.lower() for city in All]))


#%%
months = ['Januar', 'January','Februar', 'February', 'März', 'March', 'April', 'Mai', 'May', 'Juni', 'June', 'Juli', 
          'July', 'August', 'September', 'Oktober', 'October', 'November', 'Dezember', 'December']
months = [month.lower() for month in months]
print(months)


#%%
stopwords_manual = [line.rstrip('\n') for line in open('stopwords_manual.txt')]
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


#%%
stopwords_all = list(set(stopwords_germ + stopwords_eng + stopwords_manual + cities + months))
len(stopwords_all)


#%%
stemmer_own = {
    
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
    'zertifi' : 'certificate',
    'certifi' : 'certificate',
    'design' : 'design',
    'engineer' : 'engineer',
    'ingenieur'  : 'engineer',
    'entwick': 'entwicklung',
    'develop': 'entwicklung',
    'program': 'entwicklung',
    'entwickler' : 'entwicklung',
    
    'extern': 'external',
    'framework': 'framework',
    'globalen': 'global',
    'schnittstell': 'interface',
    'implement' : 'implementation', 
    'infrastr' : 'infrastructure',
    'informati' : 'informatik',
    'intern': 'internal',
    'manage' : 'management',
    'method' : 'method',
    'überwach' : 'monitoring',
    'mobil': 'mobil',
    'betrieb' : 'operation',
    
    'operat' : 'operation',
    'operie' : 'operation',
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
# Clean text and title and create new column "tokenized"
t1 = time.time()
df['tokenized'] = df['project'].apply(tokenize)
t2 = time.time()
print("Time taken to prepare", len(df), "projects documents:", (t2-t1)/60, "min")


#%%
df.head()


#%%
# Create a list containing all the words in a dataframe
all_words_df = [word for item in list(df['tokenized']) for word in item]

# Use nltk fdist to get a frequency distribution of all words
fdist_words = FreqDist(all_words_df)
print(len(fdist_words)) # number of unique words
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
print(len(df),len(train_df),len(test_df))


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


#%%
from gensim import corpora, models, similarities
# later on, load trained model from file
model =  models.LdaModel.load('LDA_Approach_1.model')

# print all topics
model.show_topics(num_topics=20, num_words=20)


#%%
import pickle
with open('dictionary', 'wb') as output:
    pickle.dump(dictionary, output)
    
with open('corpus', 'wb') as output:
    pickle.dump(corpus, output)


#%%
# Save model to disk.
from gensim.test.utils import datapath
temp_file = datapath("model")
lda.save(temp_file)

pickle.dump(lda, open('model', 'wb'))


#%%
# Load a potentially pretrained model from disk.
lda2 = LdaModel.load(temp_file)


#%%
# show_topics method shows the the top num_words contributing to num_topics number of random topics
lda.show_topics(num_topics=13, num_words=20)


#%%
lda.show_topic(topicid=0, topn=20)#ERP/SAP 


#%%
lda.show_topic(topicid=1, topn=20)# SW_Dev/Web 


#%%
lda.show_topic(topicid=2, topn=20) #IT_PM/SW_Arch 


#%%
lda.show_topic(topicid=3, topn=20)#SW_Dev/DevOps 


#%%
lda.show_topic(topicid=4, topn=20) #Sys_Admin/Support


#%%
lda.show_topic(topicid=5, topn=20) #IT_Admin/Support/Ops 


#%%
lda.show_topic(topicid=6, topn=20) #Data_Engr/Big Data


#%%
lda.show_topic(topicid=7, topn=20) #IT_Process_Mgr/Consultant


#%%
lda.show_topic(topicid=8, topn=20) #Sys_Admin/Support 


#%%
lda.show_topic(topicid=9, topn=20) #Business_Analyst/Consulting 

#%% [markdown]
# # Random project from train data

#%%
# select and article at random from train_df
#random_index = np.random.randint(len(train_df))
random_index = 1500
data_to_check = train_df.iloc[random_index,2]
#make bow frpm data_to_check
bow = dictionary.doc2bow(data_to_check)
#print(random_index)


#%%
#get description of project with index 1212
print(train_df.iloc[random_index,2])


#%%
# get the topic contributions for the document chosen above
doc_distribution = np.array([topic[1] for topic in lda.get_document_topics(bow=bow)])
doc_distribution


#%%
np.argsort(-doc_distribution)[:3]


#%%
len(doc_distribution)


#%%
# bar plot of topic distribution for this document
def plot_topic_dist(doc_distr, index):
    """
    This function plots the topic distrubtion for a given document
    It takes two parameters
    (1) doc_distr = type: list of floats, list of topic probability distribution in a document
    (2) index = type: int, index number of document to plot
    We also do 2 passes of the data since this is a small dataset, so we want the distributions to stabilize
    """
    fig, ax = plt.subplots(figsize=(12,8));
    # the histogram of the data
    patches = ax.bar(np.arange(len(doc_distr)), doc_distr)
    ax.set_xlabel('Topic ID', fontsize=15)
    ax.set_ylabel('Topic Probability Score', fontsize=15)
    ax.set_title("Topic Distribution for Project in Index " + str(index), fontsize=20)
    ax.set_xticks(range(0,10))
    x_ticks_labels = ['ERP/SAP','SW_Dev/Web','IT_App_Mgr/SW_Dev_Arch','SW_Dev/DevOps','Sys_Admin/Support', 'IT_Admin_SW/Oracle/Ops','Data/Ops','IT_Process_Mgr/Consultant', 'MS_DEV/Admin','Business_Analyst/Consulting']
    ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
    fig.tight_layout()
    return plt.show()


#%%
plot_topic_dist(doc_distribution, random_index)


#%%
# print the top 5 contributing topics and their words
#for i in doc_distribution.argsort()[-5:][::-1]:
    #print(i, lda.show_topic(topicid=i, topn=10), "\n")


#%%
lda_model =  models.LdaModel.load('lda.model')


#%%
lda_model.show_topics()


#%%
doc_distribution1 = np.array([topic[1] for topic in lda_model.get_document_topics(bow=bow2)])
labels = np.argmax(doc_distribution1)
print(doc_distribution1)

#%% [markdown]
# # New Unseen Test Data

#%%
doc_index = 21


#%%
new_doc_bow = dictionary.doc2bow(test_df.iloc[doc_index,3])


#%%
print(test_df.iloc[doc_index,1])


#%%
print(test_df.iloc[doc_index,2])


#%%
new_doc_distr = np.array([topic[1] for topic in lda.get_document_topics(bow=new_doc_bow)])


#%%
new_doc_distr


#%%
np.sum(new_doc_distr)


#%%
def percentage(data):
    total = np.sum(data)
    perc_arr = np.array([(x/total)*100 for x in data])
    return perc_arr
        


#%%
percentage(new_doc_distr).argmax()


#%%
plot_topic_dist(new_doc_distr, doc_index)


#%%
# print the top 5 contributing topics and their words
for i in new_doc_distr.argsort()[-5:][::-1]:
    print(i, lda.show_topic(topicid=i, topn=10), "\n")

#%% [markdown]
# # Finding similar projects

#%%
#list of topics distribution in tuples among all documents
#lda[corpus] => genism_lda bow dictionary
all_topic_distr_list = lda[corpus]


#%%
#get each topic probablity distr and convert to an array
#simply a list each documents topic distribution
corpus_topic_dist= np.array([[topic[1] for topic in docs] for docs in all_topic_distr_list])
corpus_topic_dist[:3]


#%%
#Jensen Shannon Distance calculates the statistical similarity between two documents. 
#The smaller the value , the closer or similare both documents are
#Its symmetric (associative) meaning the similarity value btw A & B is the same btw B & A

def js_similarity_score(doc_distr_query, corpus_distr):
    """
    This function finds the similarity score of a given doc accross all docs in the corpus
    It takes two parameters: doc_distr_query and corpus_distr
    (1) doc_distr_query is the input document query which is an LDA topic distr: list of floats (series)
            [1.9573441e-04,...., 2.7876711e-01]
    (2) corpus_dist is the target corpus containing the LDA topic distr of all documents in the corpus: lists of lists of floats (vector)
            [[1.9573441e-04, 2.7876711e-01, 1.9573441e-04]....[1.9573441e-04,...., 2.7876711e-01]]
    It returns an array containing the similarity score of each document in the corpus_dist to the input doc_distr_query
    The output looks like this: [0.3445, 0.35353, 0.5445,.....]
    
    """
    input_doc = doc_distr_query[None,:].T #transpose input
    corpus_doc = corpus_distr.T # transpose corpus
    m = 0.5*(input_doc + corpus_doc)
    sim_score = np.sqrt(0.5*(entropy(input_doc,m) + entropy(corpus_doc,m)))
    return sim_score


#%%
def find_top_similar_docs(doc_distr_query, corpus_distr,n=10):
    """
    This function returns the index lists of the top n most similar documents using the js_similarity_score
    n can be changed to any amount desired, default is 10
    """
    sim_score = js_similarity_score(doc_distr_query, corpus_distr)
    similar_docs_index_array = sim_score.argsort()[:n] #argsort sorts from lower to higher
    return similar_docs_index_array


#%%
# select and article at random from test_df
random_article_index = np.random.randint(len(test_df))
print(random_article_index)


#%%
with open('traindata.sav', 'wb') as output:
    pickle.dump(train_df, output)


#%%
appdata = train_df[['title', 'description']]
appdata.head()


#%%
with open('APP_DATA.sav', 'wb') as output:
    pickle.dump(appdata, output)


#%%
def recommend(text):
    clean = all_processing(text)
    text_bow = dictionary.doc2bow(clean)
    new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=text_bow)])
    corpus_topic_dist= np.array([[topic[1] for topic in docs] for docs in all_topic_distr_list])
    similar_docs_index = find_top_similar_docs(new_doc_distribution, corpus_topic_dist)
    top_sim_doc = train_df[train_df.index.isin(similar_docs_index)]
    PROJECT_DICT = top_sim_doc.to_dict() 
    return PROJECT_DICT
    


#%%
new_proj = """
Für unseren Kunden aus dem Bankenumfeld in Frankfurt am Main sind wir aktuell auf der Suche nach einem Full-Stack Java Entwickler (m/w/d) im Bereich CI/CD . 

Falls Sie die folgende Projektbeschreibung interessiert und Sie die Anforderungen ausreichend abdecken, bitten wir Sie um kurzfristige Rückmeldung unter Angabe Ihrer Kontaktdaten, Ihrer Konditionen für den genannten Einsatzort (Stunden- oder Tagessatz) sowie Ihres Profils (vorzugsweise in Word).
Gerne können Sie uns dieses per E-Mail an schicken. Da der E-Mailversand bekanntermaßen nicht zu den sichersten Datenübertragungen zählt, möchten wir Ihnen zusätzlich eine sichere und verschlüsselte Upload-Möglichkeit für Ihre Bewerbungsunterlagen anbieten. Nutzen Sie dazu die Schaltfläche „Bewerben“ in unserem Projektportal unter https://mindheads.de.

Projektstandort: Frankfurt am Main
Dauer: 
Abgabefrist beim Kunden: zeitnah

Kurzbeschreibung Projekt:
 
Als Software Engineer / Java Fullstack Developer (m/w/d) im CI-CD & Agile QA-Cluster sind Sie an der Einrichtung, Entwicklung und Verwaltung von Softwareprozessen innerhalb der Systeme Continuous Delivery und Continuous Integration des Kunden beteiligt. Sie unterstützen die Einführung, Wartung und Betreuung neuer und bestehender Produkte im Software-Lebenszyklus. Die Zusammenarbeit basiert auf agilen Methoden und besteht aus einem Team von Business Analysten, Softwarearchitekten, Scrum Mastern und anderen Spezialisten mit unterschiedlichem Hintergrund.

Wichtig: Der Kandidat muss fest angestellt sein! 
 
Aufgaben:
Weiterentwicklung des CI/CD Clusters 
Übertragung von Konzepten, Designs und Architekturmodelle in die Micro-Service-Architektur 
Design und Implementierung von REST-APIs
Design und Implementierung von Web-Frontends 

Anforderungen:
Fundierte Erfahrung in der Full-Stack Java / Java EE-Entwicklung mit Schwerpunkt auf Micro Service Architektur,
Fundierte Erfahrung in agilen Arbeitsmethoden.
Expertise in Web-Frontends z.B. React, Angular
Umfangreiche Erfahrung mit Spring Boot
Umfangreiche Erfahrung mit RDBMS (Oracle) und SQL
Sehr gute Kenntnisse von SSL / TLS (Kerberos, SAML, OAuth, etc.), HTTP-Protokollen, UML und Design Patterns
Sehr gute Kenntnisse in Wort und Schrift Deutsch und Englisch
Erfahrung in der Cloud-Entwicklung (Openshift) wünschenswert

 

Für Fragen, Anregungen oder Wünsche stehen wir Ihnen gern zur Verfügung. Aktuelle Informationen über uns sowie weitere Vakanzen finden Sie auch auf unserer Homepage: https://mindheads.de
"""


#%%
pr = recommend(new_proj)


#%%
pr.title[595]


#%%
PROJECT_DICT = dict(zip(train_df['title'], train_df['description']))
PROJECT_DICT


#%%
new_bow = dictionary.doc2bow(test_df.iloc[random_article_index,3])
new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])


#%%
a = ""


#%%
new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=b)])


#%%
# this is surprisingly fast
similar_docs_index = find_top_similar_docs(new_doc_distribution, corpus_topic_dist)


#%%
similar_docs_index


#%%
top_sim_docs = train_df[train_df.index.isin(similar_docs_index)]
top_sim_docs['title']


#%%
def predict_bereich(text, lda_model):
    clean = all_processing(text)
    text_bow = dictionary.doc2bow(clean)
    topic_distr_array = np.array([topic[1] for topic in lda_model.get_document_topics(bow=text_bow)])
    plot_topic_dist(topic_distr_array, 1)
    #return topic_distr_array
    for i in topic_distr_array.argsort()[-5:][::-1]:
        print(i, lda_model.show_topic(topicid=i, topn=10), "\n")
    return topic_distr_array, clean, text_bow
    


#%%
t, c, b = predict_bereich(te, lda_model)


#%%
t


#%%
check1 = dictionary.doc2bow(t)


#%%
c


#%%



#%%
def clean_lower_tokenize(text):
    """
    Function to clean, lower and tokenize texts
    Returns a list of cleaned and tokenized text
    """
    #text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)  #remove websites texts like email, https, www
    text = re.sub("[^a-zA-Z ]", "", text) #remove non alphbetic text
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text


#%%
def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stopwords]


#%%
def stem_eng_german_words(text):
    """
    Function to stem words
    """
    try:
        text = [stemmer_germ.stem(word) for word in text]
        #text = [stemmer_eng.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] 
    except IndexError:
        pass
    return text


#%%
def all_processing(text):
    """
    This function applies all the functions above into one
    """
    return stem_eng_german_words(remove_stop_words(clean_lower_tokenize(text)))


