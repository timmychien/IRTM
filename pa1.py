from urllib.request import urlopen 
import string
from nltk.stem.porter import PorterStemmer

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#read file
news=urlopen('https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt')
news=news.read().decode('utf-8')

#Lowercase
news=news.lower()

#tokenize
def tokenize(article):
  for punctuaction in string.punctuation:
    article=article.replace(punctuaction,'')
  article=article.replace('\r\n','')
  article=article.split(' ')
  return article

news=tokenize(news)

#stemming
stemmer=PorterStemmer()
stemlist=[]
for token in news:
  stemlist.append(stemmer.stem(token))

#stopword removal
stopwords=urlopen('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words')
stopwords=stopwords.read().decode('utf-8')
stopwords=stopwords.replace('\r\n',' ')
stopwords=stopwords.split(' ')
result=[token for token in stemlist if token not in stopwords]

#save txt file
result=" ".join(result)
output=open('result.txt','wb')
output.write(result.encode())
output.close()