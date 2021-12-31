from urllib.request import urlopen
import requests
import string
import re
from nltk.stem.porter import PorterStemmer
import ssl
import math
import pandas as pd
ssl._create_default_https_context = ssl._create_unverified_context
def tokenize(article):
    article=article.lower()
    for punctuaction in string.punctuation:
        article=article.replace(punctuaction,'')
    article=re.sub(r'\b(?:\d+|\w)\b\s*','',article)
    article=re.sub(r'[0-9]','',article)
    article=article.replace('\r','')
    article=article.replace('\n','')
    article=article.split(' ')
    return article
def stopword(stopwordlist,article):
    stop_result=[token for token in article if token not in stopwordlist]
    return stop_result
def stemming(stopword_result):
    stemmer=PorterStemmer()
    stemlist=[]
    for token in stopword_result:
      stemmed_token=stemmer.stem(token)
      stemlist.append(stemmed_token)
      result=" ".join(stemlist)
    return result
def create_dictionary(docsN):
    diction={}
    for doc in docsN:
      for word in set(doc.split(' ')):
        if word not in diction:
          diction[word]=1
        else:diction[word]+=1
    return diction
def select_feature(k,dictN):
  features=[]
  dict_sorted=sorted(dictN.items(), key=lambda x:x[1])
  term_list=[terms[0] for terms in dict_sorted if len(terms[0])>3]
  features=term_list[(len(term_list)-k):len(term_list)]
  return features
def trainMultinomialNB(C,dataset):
  T=dict()
  condprob=dict()
  docsN=dict()
  dictN=dict()
  V=list()
  prior=dict()
  #ExtractVocabulary + feature selection
  for j in range(1,len(dataset)+1):
    docsN[j]=list()
    dictN[j]=dict()
    for idx in range(len(dataset[j])):
      docid=dataset[j][idx]
      docsN[j].append(documents[docid-1])
    dictN[j]=create_dictionary(docsN[j])
  #CountDocs
  N=0
  for c in C:
    N+=len(dataset[c])
    Nc=len(dataset[c])
    V+=select_feature(25,dictN[c])
    prior[c]=Nc/N
  #CountDocsInClass
  V=set(V)
  for c in C:
    T[c]=dict()
    condprob[c]=dict()
    #ConcatenateTextofAllDocsInClass
    text=''
    for doc in docsN[c]:
      text=text+' '+doc
    for t in V:
      if t in text.split(' '):
        T[c][t]=text.count(t)
      else:T[c][t]=0
      #CountTokensofTerms
      condprob[c][t]=(T[c][t]+1)/(sum(T[c].values())+len(V))
  return V,prior,condprob
def applyMultinomialNB(C,V,prior,condprob,d):
  W=[]
  score={}
  #ExtractTokenFromDoc
  for token in V:
    if token in d.split(' '):
      W.append(token)
  for c in C:
    score[c]=math.log(prior[c])
    for t in W:
      if t in condprob[c]:
        score[c]+=math.log(condprob[c][t])
  return max(score,key=score.get)
stopwords=urlopen('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words')
stopwords=stopwords.read().decode('utf-8')
stopwords=stopwords.replace('\r\n',' ')
stopwords=stopwords.split(' ')
stopwords.remove('')
add_stop=['about','accordance','according','accordingly','along','also','been','away','better','begin','began','came','contain','especially','good','home','involving','just','look','necessarily','obviously','possible','possibly','said','seven','something','stop','dont','along','also','been','come','could','still','take','than','that','them','then','there','these','those','three','told','under','until','well','were','what','from','always','begun','eight','have','is','into','last','made','most','other','possible','should',
          'stop','absolutely','allow','contains','containing','given','immediately','include','includes','including','later','thing','today','tomorrow','think','recently','known','greater','associated','associating','value','took','take','like','despite','actually','change','happens','keep','keeps']
for add in add_stop:
  stopwords.append(add)
training=requests.get('https://ceiba.ntu.edu.tw/course/88ca22/content/training.txt')
training_txt = open('training.txt', 'w')
training_txt.write(training.text)
training_txt.close()
train=pd.read_csv('./training.txt',names=['class_id','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'],delim_whitespace=True)
training_set={}
for c in train['class_id']:
  training_set[c]=train.iloc[c-1][1:]
#Preprocessing
documents=[]
for num in range(1,1096):
  f=open('./IRTM/'+str(num)+'.txt','r').read()
  tokens=tokenize(f)
  ls=stopword(stopwords,tokens)
  results=stemming(ls)
  documents.append(results)
C=[c for c in range(1,14)]
#training phase
V_train,prior_train,condprob_train=trainMultinomialNB(C,training_set)
#testing phase
testing_set=[idx for idx in range(1,1096)]
for i in range(1,len(training_set)+1):
  for j in range(len(training_set[i])):
    testing_set.remove(training_set[i][j])
result_=list()
for i in testing_set:
  result_.append(applyMultinomialNB(C,V_train,prior_train,condprob_train,documents[i-1]))
result_csv=pd.DataFrame(result_,columns=['Value'])
result_csv['Id']=testing_set
result_csv=result_csv.set_index(['Id'])
result_csv.to_csv("r10725057_pa3.csv")