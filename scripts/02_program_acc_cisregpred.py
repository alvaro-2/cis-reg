import urllib3
import urllib
import pandas as pd

def submit(acc, email):
    email = email.replace('@', '%40')
    url = "http://aimpred.cau.ac.kr/fetchBatch?uniprotid=" + acc + "&email="+email
    req = urllib.request.urlopen(url)
    txt = req.read()    

    return txt
 

def retrieve(acc, email):
    email = email.replace('@', '%40')
    url = "http://aimpred.cau.ac.kr/fetchResult?uniprotid="+acc+"&email="+email
    req = urllib.request.urlopen(url)
    txt = req.read()
    
    return txt
    
# Now bring a list of proteins (as UniProt accesion)
protein = pd.read_csv('../datasets/protein.tsv', sep = '\t')
proteins = protein.uniprot_acc[:50].to_list()

# Code for submission
email = "amnalvaro@gmail.com"

for p in proteins:
    print(p, '\t', submit(p, email))

# Code for result retrieval
for p in proteins:
    r = retrieve(p, email)

    if len(r.strip()) == 0:
        print(p, 'Result is not available')
    else:
        print(p, 'saved into a file')
        #p = p.encode('utf-8')
        f=open(p+'.txt','w')
        #f.readlines()
        #print(f.read())
        f.write(r.replace('<br>',''))
        f.close()