import nltk
#Using NLTK Downloader to obtain the resource stopwords, punkt
nltk.download('stopwords')
nltk.download('punkt')

currDir = os.getcwd()
if "USL" not in currDir:
    os.chdir(os.path.join(currDir,  "ML", "USL"))
else:
    os.chdir("../USL")

# load nltk's German and english stopwords'
dataDir = os.path.join(currDir,  "constants")
with open(os.path.join(dataDir, 'german_stopwords_full.txt'), 'r') as f:
    stopwords_germ = f.read().splitlines()
stopwords_eng = nltk.corpus.stopwords.words('english')
combined_stopWordsPath = os.path.join(dataDir, 'stopwords_manual.txt')

#german cities
from constants.bundeslander import Baden_Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg, Hessen, Mecklenburg_Vorpommern, Niedersachsen, Nordrhein_Westfalen, Rheinland_Pfalz, Saarland, Sachsen, Sachsen_Anhalt, Schleswig_Holstein, Thüringen, Ausland

All = Baden_Württemberg + Bayern + Berlin + Brandenburg + Bremen +Hamburg + Hessen + Mecklenburg_Vorpommern + Niedersachsen + Nordrhein_Westfalen + Rheinland_Pfalz + Saarland + Sachsen + Sachsen_Anhalt + Schleswig_Holstein + Thüringen + Ausland
cities = list(set([city.lower() for city in All]))

months = ['Januar', 'January','Februar', 'February', 'März', 'March', 'April', 'Mai', 'May', 'Juni', 'June', 'Juli', 
          'July', 'August', 'September', 'Oktober', 'October', 'November', 'Dezember', 'December']
months = [month.lower() for month in months]
#print(months)

stopwords_manual = [line.rstrip('\n') for line in open(combined_stopWordsPath)]
#print(len(stopwords_manual))

stopwords_all = list(set(stopwords_germ + stopwords_eng + stopwords_manual + cities + months))
len(stopwords_all)

stopwordsLocation = os.path.join(dataDir,  "stopwords.pickle")
pickle_out = open(stopwordsLocation,"wb")
pickle.dump(stemmer_own, pickle_out)
pickle_out.close()