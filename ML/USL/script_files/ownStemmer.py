import pickle

stemmer_own = {
    
    'admin': 'admin',  
    'verwaltung': 'administration',
    'architektur' : 'architektur',
    'agil' : 'agile',
    'app': 'application',
    'anwend' : 'application',
    'automat': 'automate',
    'consultant' : 'berater',
    'berat': 'berater',
    'cisco': 'cisco',
    'contin': 'continuous',
    'schnittstell': 'interface',
    'überwach' : 'monitoring',
    'mobil': 'mobil',
    
}

pickle_out = open("../constants/stemmer_own.pickle","wb")
pickle.dump(stemmer_own, pickle_out)
pickle_out.close()