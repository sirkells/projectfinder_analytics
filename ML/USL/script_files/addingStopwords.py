import os
import nltk
import pickle

def getStopWords(currDir):
    dataDir = os.path.join(currDir,  "constants")
    stopwordsLocation = os.path.join(dataDir, "stopwords.pickle")
    pickle_in = open(stopwordsLocation,"rb")
    stopwords = pickle.load(pickle_in)
    #print(len(stopwords))
    return stopwords
   
    
def addingStopWords(words, stopwords):
    stopwords.extend(words)
    #print(len(stopwords))
    newStopWords = list(set(stopwords))
    return newStopWords

def saveNewStopWords(currDir, newStopWords):
    dataDir = os.path.join(currDir,  "constants")
    stopwordsLocation = os.path.join(dataDir,  "stopwords.pickle")
    pickle_out = open(stopwordsLocation,"wb")
    pickle.dump(newStopWords, pickle_out)
    pickle_out.close()
    
def modifyingStopWords(newWords):
    os.chdir('..')
    currDir = os.getcwd()
    print(currDir)
    stopWords = getStopWords(currDir)
    #print(len(stopWords))
    #print(len(newWords))
    newStopWords = addingStopWords(newWords, stopWords)
    saveNewStopWords(currDir, newStopWords)
    print('Added new StopWords')
    
    
    
if __name__ == "__main__":
    newWords = ['hochschulabschluss', 'fachhochschulabschluss']
    modifyingStopWords(newWords)