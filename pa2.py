from urllib.request import urlopen
import string
from nltk.stem.porter import PorterStemmer
import ssl
import os
import math
import numpy as np
import collections
ssl._create_default_https_context = ssl._create_unverified_context
def tokenize(article):
    article=article.lower()
    for punctuaction in string.punctuation:
        article=article.replace(punctuaction,'')
    article=article.replace('\r','')
    article=article.replace('\n','')
    article=article.split(' ')
    return article
def stemming(article):
    stemmer=PorterStemmer()
    stemlist=[]
    for token in article:
        stemlist.append(stemmer.stem(token))
    return stemlist
def stopwords(stemlist):
    stopwords=urlopen('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words')
    stopwords=stopwords.read().decode('utf-8')
    stopwords=stopwords.replace('\r\n',' ')
    stopwords=stopwords.split(' ')
    result=[token for token in stemlist if token not in stopwords]
    result=" ".join(result)
    return result

def create_dictionary(documents):
    diction={}
    dictionary=[]
    for doc in documents:
        for word in set(doc.split(' ')):
            if word not in diction:
                diction[word]=1
            else:diction[word]+=1
    items=diction.items()
    items=sorted(items)
    for i in range(0,len(diction)):
        dictionary.append([i+1,items[i]])
    return dictionary
def tfidf(docId):
    global dictionary
    global documents
    document=documents[docId-1]
    idf={}
    tf={}
    tf_idx={}
    tfidf={}
    for word in document.split(' '):
        for i in range(len(dictionary)):
            if word==dictionary[i][1][0]:
                if word not in tf:
                    tf[word]=1
                else:tf[word]+=1
    for word in tf:
        for i in range(len(dictionary)):
            if word==dictionary[i][1][0]:
                tf_idx[dictionary[i][0]]=tf[word]
    for i in range(len(dictionary)):
        idf[dictionary[i][0]]=math.log10(1095/int(dictionary[i][1][1]))
    for idx in tf_idx:
        tfidf[idx]=(int(tf_idx[idx])*idf[idx])/len(tf_idx)
    tfidf=collections.OrderedDict(sorted(tfidf.items()))
    return tfidf
def cosine(Docx,Docy):
    global dictionary
    tfidf1=tfidf(Docx)
    tfidf2=tfidf(Docy)
    diff1=tfidf1.keys()-tfidf2.keys()
    diff2=tfidf2.keys()-tfidf1.keys()
    for key in diff1:
        tfidf2[key]=0
    for key in diff2:
        tfidf1[key]=0
    #dict to sorted tuple
    tfidf1_items = tfidf1.items()
    tfidf1_items=sorted(tfidf1_items)
    tfidf2_items = tfidf2.items()
    tfidf2_items=sorted(tfidf2_items)
    v1=[tfidf1_items[i][1] for i in range(len(tfidf1_items))]
    v2=[tfidf2_items[j][1] for j in range(len(tfidf2_items))]
    v1=np.array(v1)
    v2=np.array(v2)
    sim=(v1.dot(v2))
    return sim

documents=[]
#preprocessing

all_files=os.listdir('./IRTM')
for file in all_files:
    f=open('./IRTM/'+file,'r').read()
    f=tokenize(f)
    ls=stemming(f)
    result=stopwords(ls)
    output=open('./data/'+file,'wb')
    output.write(result.encode())

all_files_stemmed=os.listdir('./data')
for file in all_files_stemmed:
    f=open('./data/'+file,'r').read()
    documents.append(f)
#1.create dictionary.txt
dictionary=create_dictionary(documents)
dictionary_output=open('dictionary.txt','wb')
dictionary_output.write(f't_index\tterm\tdf\n'.encode())
for i in range(len(dictionary)):
    dictionary_output.write(f'{dictionary[i][0]}\t{dictionary[i][1][0]:^}\t{dictionary[i][1][1]:^}\n'.encode())
dictionary_output.close()

#2.tf-idf 1.txt
tfidf1=tfidf(1)
tfidf1=list(tfidf1.items())
tfidf1_output=open('doc1.txt','wb')
tfidf1_output.write(f'amount of term:\n'.encode())
tfidf1_output.write(f'{len(tfidf1)}\n'.encode())
tfidf1_output.write(f't_index\ttf-idf\n'.encode())
for i in range(len(tfidf1)):
    tfidf1_output.write(f'{tfidf1[i][0]}\t{tfidf1[i][1]:^}\n'.encode())
tfidf1_output.close()

# tfidf of all documents

for idx in range(len(documents)):
    tfidf_doc=tfidf(i)
    tfidf_doc=list(tfidf_doc.items())
    tfidf_output=open(f'./output/{idx+1}.txt','wb')
    tfidf_output.write(f't_index\ttf-idf\n'.encode())
    for i in range(len(tfidf_doc)):
        tfidf_output.write(f'{tfidf_doc[i][0]}\t{tfidf_doc[i][1]}\n'.encode())
    tfidf_output.close()

#3.calculate cosine simularity
cosine1_2=cosine(1,2)
print('cosine similaritiy of Doc1 and Doc2:',cosine1_2)

    