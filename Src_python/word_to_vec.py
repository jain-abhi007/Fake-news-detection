# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:01:38 2019

@author: ashu_
"""

# -*- coding: utf-8 -*-
import math
import gensim.downloader as api
from gensim.models import Word2Vec
import numpy as np
from scipy import spatial
import string
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib
filename='word_to_vect.sav'
model=joblib.load(filename)
stopWords=['be','which', 'y', 'myself', 'our', 'doing', 'isn', 'ma', 'o', 'yourself', 's', 'itself', 'ourselves', 'does', 'has', 'my', 'll', 'he', 'own', 'or', 'having', 'mustn', 'at', 'too', 'herself', 'other', 'themselves', 'very', 'don', 'me', 'these', 'with', "she's", 'can', 'is', 'off', 'in', 'to', 'shan', 'those', 'most', 'himself', 'them', 'there', 'ain', 'hasn', 'their', 'nor', 've', 'she', 'was', 'hadn', 'being', 'both', "it's", 'just', 'up', 'as', 'wouldn', 'aren', 'some', 'his', 'we', 'same', 'and', 'more', 'ours', 'because', 'mightn', 'of', 'will', 'do', 'on', 'are', 'no', 'if', "you're", 't', 'about', 'so', 'after', 'few', 'had', 'yourselves', 'while', 'd', 'over', 'this', 'any', 'its', 'once', 'that', 'a', 'again', 'how', 'it', 'who','than', "you'd", 'but', 'until', 'each', 'why', "you'll", 'you', 'from', 'further', 'an', 'through', 'yours', 'have', 'into', 'your', 'should', "mightn't", 'all', 'were', 'by', 're', 'been', 'hers', 'haven', 'him', "that'll", 'during',  'down', 'they', 'out', "should've", 'theirs', 'm', 'the', 'whom', 'when', 'what', 'did', 'her', 'here', 'where', "you've", 'am', "shan't", 'only', 'such', "mustn't", 'then', 'needn', "hadn't", 'weren', 'under', 'i', 'for']
vocab=list(model.vocab)
def extract():
    import requests
    from bs4 import BeautifulSoup
    import csv
    global abc
    print(abc)
    headline=abc
    headline=headline.replace(' ','+')
    lk='https://www.google.com'
    link='https://www.google.com/search?q={}&num={}&hl={}'.format(headline,10,'en')
    USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    csv_file=open("body.csv",'w',encoding="utf-8")
    writer=csv.writer(csv_file)
    header=["Body"]
    writer.writerow(header)
    pa="Page "
    for i in range(1,5):
        page=requests.get(link,headers=USER_AGENT)
        page.raise_for_status()
        soup=BeautifulSoup(page.text,'lxml')
        bodys=soup.select(".st")
        for body in bodys:
            #print(body.get_text())
            writer.writerow([body.get_text()])
        att=pa+str(i+1)
        next=(soup.find_all(attrs={"aria-label":att})[0])["href"]
        link=lk+next
    
def remove_stopwords(data):
    wordsFiltered=[]
    global stopWords
    words=word_tokenize(data)
    for w in words:
        w=w.lower()
        if w not in stopWords:
            wordsFiltered.append(w)
    table=str.maketrans('','',string.punctuation)
    for i in range(len(wordsFiltered)):
        wordsFiltered[i]=wordsFiltered[i].translate(table)
    ans=""
    for i in range(len(wordsFiltered)):
        ans=ans+" "+wordsFiltered[i]
    return ans

def avg_feature_vector(sentence, model, num_features, vocab):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in vocab:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

while True:
    ans=0.0
    max_res=0
    min_res=1
    import csv
    global abc
    with open('data.csv','r',encoding="utf-8") as cf:
        re=csv.reader(cf)
        md=[]
        global abc
        abc=""
        for row in re:
            if len(row)==0:
                continue
            md=row
        for a in md:
            abc=abc+a
    abc=input("enter the statement :")
    sent1=remove_stopwords(abc)
    extract()
    count=0
    with open('body.csv','r',encoding="utf-8") as csv_file:
        reader=csv.reader(csv_file)
        for row in reader:
            if len(row)==0:
                continue
            data=row[0]
            sent2=remove_stopwords(data)
            s1_afv = avg_feature_vector(sent1, model=model, num_features=300, vocab=vocab)
            s2_afv = avg_feature_vector(sent2, model=model, num_features=300, vocab=vocab)
            sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
            if math.isnan(sim) != True:
                if sim>max_res:
                    max_res=sim
                if sim<min_res:
                    min_res=sim
                count=count+1
                if count==5:
                    break
    print(max_res,min_res)
    flag=input("\n do you want to continue ? yes=1,no=0")
    if flag=="0":
        break
print("Thank You")
