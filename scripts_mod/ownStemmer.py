import pickle

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

pickle_out = open("stemmer_own.pickle","wb")
pickle.dump(stemmer_own, pickle_out)
pickle_out.close()