'''
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwordls=stopwords.words('english')
from nltk.stem.wordnet import WordNetLemmatizer
'''
from urllib.request import urlopen
import requests
import os
import string
import re
from nltk.stem.porter import PorterStemmer
import collections
import ssl
import math

import pandas as pd
import numpy as np
#ssl._create_default_https_context = ssl._create_unverified_context
#preprocess
def tokenize(article):
  article=article.replace('\r','')
  article=article.replace('\n','')
  article=re.sub(r'\b(?:\d+|\w)\b\s*','',article)
  article=re.sub(r'[0-9]','',article)
  article=article.lower()
  for punctuaction in string.punctuation:
    article=article.replace(punctuaction,'')
  tokens=article.split(' ')
  return tokens
def stopword(tokens,stopwordlist):
  stop_result=[token for token in tokens if token not in stopwordlist]
  return stop_result
def stemming(stopword_result):
  stemmer=PorterStemmer()
  stemlist=[]
  for token in stopword_result:
    stemmed_token=stemmer.stem(token)
    stemlist.append(stemmed_token)
  return stemlist
def lemmatize(stemming_result):
  lemmatizelist=[]
  lemmatizer = WordNetLemmatizer()
  for token in stemming_result:
    lemmatized_token=lemmatizer.lemmatize(token)
    lemmatizelist.append(lemmatized_token)
    lemmatize_result=" ".join(lemmatizelist)
  return lemmatize_result
#create dictioinary
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
#tfidf
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
#cosine similarity
def cosine(Docx,Docy):
    tfidf1=Docx
    tfidf2=Docy
    union_index=tfidf1.index.union(tfidf2.index)
    tfidf1_index=[i for i in union_index if i not in tfidf1.index]
    tfidf2_index=[i for i in union_index if i not in tfidf2.index]
    tfidf1_=pd.DataFrame([0]*len(tfidf1_index),index=tfidf1_index,columns=['tf-idf'])
    tfidf2_=pd.DataFrame([0]*len(tfidf2_index),index=tfidf2_index,columns=['tf-idf'])
    tfidf1=pd.concat([tfidf1_,tfidf1]).sort_index()
    tfidf2=pd.concat([tfidf2_,tfidf2]).sort_index()
    v1=np.array(tfidf1['tf-idf'])
    v2=np.array(tfidf2['tf-idf'])
    sim=v1.dot(v2)
    return sim
#HAC
def HAC(C,clusters):
  I=[0]*(len(C))
  A=list()
  for n in range(len(C)):
    I[n]=1
  for k in range(len(C)-1):
    if I.count(1)==clusters:
      break
    a,b=np.unravel_index(C.argmax(),C.shape)
    i,m=min(a,b),max(a,b)
    A.append([i,m])
    for j in range(len(C)):
      if i==j:
        C[i][j]=0
        C[j][i]=0
      else:
        if C[i][j]<C[m][j] and C[i][j]>0:
          C[i][j]=C[i][j]
        if C[i][j]>C[m][j] and C[m][j]>0:
          C[i][j]=C[m][j]
        if C[j][i]<C[j][m] and C[j][i]>0:
          C[j][i]=C[j][i]
        if C[j][i]>C[j][m] and C[j][m]>0:
          C[j][i]=C[j][m]
    I[m]=0
    C[m]=0
    for idx in range(len(C)):
      C[idx][m]=0
  for i in range(len(C)):
    A.append([i,i])
  return A
# merge
def merge(diction):
  key_remove=[]
  keys=[]
  for key in diction:
    keys.append(key)
  for key in keys:
    for ele in diction[key]:
      if ele in diction and ele!=key:
        diction[key]=diction[key]+diction[ele]
        del diction[ele]
        keys.remove(ele)
  return diction
#output
def output(ans,clusters):
  c_dict=dict()
  key_remove=[]
  for i in range(len(ans)):
    if ans[i][0]+1 not in c_dict:
      c_dict[ans[i][0]+1]=list()
      c_dict[ans[i][0]+1].append(ans[i][1]+1)
    else:c_dict[ans[i][0]+1].append(ans[i][1]+1)
  for key in c_dict:
    c_dict[key].sort()
  while len(c_dict)>clusters:
    c_dict=merge(c_dict)
  for key in c_dict:
    c_dict[key]=set(c_dict[key])
    c_dict[key]=list(c_dict[key])
    c_dict[key].sort()
  return c_dict
#preprocessing
stopwordls=stopwords.words('english')
documents=[]
for num in range(1,1096):
  f=open('./IRTM/'+str(num)+'.txt','r').read()
  tokens=tokenize(f)
  ls=stopword(tokens,stopwordls)
  stemmedlist=stemming(ls)
  results=lemmatize(stemmedlist)
  documents.append(results)
#tf-idf
for idx in range(1,len(documents)+1):
  tfidf_doc=tfidf(idx)
  tfidf_doc=list(tfidf_doc.items())
  tfidf_output=open(f'./output/{idx}.txt','wb')
  tfidf_output.write(f't_index\ttf-idf\n'.encode())
  for i in range(len(tfidf_doc)):
    tfidf_output.write(f'{tfidf_doc[i][0]}\t{tfidf_doc[i][1]}\n'.encode())
  tfidf_output.close()
#cosine similarity
tfidfs=[]
for i in range(1,1096):
  _doc=pd.read_csv(f'./output/{str(i)}.txt', sep="\t")
  _doc=_doc.set_index(['t_index'])
  tfidfs.append(_doc)
Cosines=np.zeros(shape=(len(tfidfs),len(tfidfs)))
for n in range(len(tfidfs)):
  for i in range(len(tfidfs)):
    if n==i:
      Cosines[n][i]=0
    else:
      Cosines[n][i]=cosine(tfidfs[n],tfidfs[i])
col=[i for i in range(1,1096)]
idx=[i for i in range(1,1096)]
df=pd.DataFrame(Cosines,index=idx,columns=col)
df.to_csv('./sim.csv')
#8 clusters
df=pd.read_csv('./sim.csv')
df=df.drop(['Unnamed: 0'],axis=1)
Cosines=df.to_numpy()
cos_8=Cosines
ans_8=HAC(cos_8,8)
cluster_8=output(ans_8,8)
cluster8_key=[]
for key in cluster_8:
  cluster8_key.append(key)
cluster8_key.sort()
cluster8_out=open('./8.txt','wb')
for key in cluster8_key:
  for ele in cluster_8[key]:
    cluster8_out.write(f'{ele}\n'.encode())
  cluster8_out.write('\n'.encode())
cluster8_out.close()

#13 clusters
df=pd.read_csv('./sim.csv')
df=df.drop(['Unnamed: 0'],axis=1)
Cosines=df.to_numpy()
cos_13=Cosines
ans_13=HAC(cos_13,13)
cluster_13=output(ans_13,13)
cluster13_key=[]
for key in cluster_13:
  cluster13_key.append(key)
cluster13_key.sort()
cluster13_out=open('./13.txt','wb')
for key in cluster13_key:
  for ele in cluster_13[key]:
    cluster13_out.write(f'{ele}\n'.encode())
  cluster13_out.write('\n'.encode())
cluster13_out.close()

#20 clusters
df=pd.read_csv('./sim.csv')
df=df.drop(['Unnamed: 0'],axis=1)
Cosines=df.to_numpy()
cos_20=Cosines
ans_20=HAC(cos_20,20)
cluster_20=output(ans_20,20)
cluster20_key=[]
for key in cluster_20:
  cluster20_key.append(key)
cluster20_key.sort()
cluster20_out=open('./20.txt','wb')
for key in cluster20_key:
  for ele in cluster_20[key]:
    cluster20_out.write(f'{ele}\n'.encode())
  cluster20_out.write('\n'.encode())
cluster20_out.close()
