# -*- coding: utf-8 -*-
"""
===== Final-Usage Model =====
"""
print(__doc__)

from sklearn.externals import joblib
import numpy as np
import csv
from warnings import simplefilter
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import math
from scipy import spatial
import string
import requests
from warnings import simplefilter

filename_word2vec='text8.sav'
model_word2vec=joblib.load(filename_word2vec)
vocab=list(model_word2vec.wv.vocab)
stopWords=['be','which', 'y', 'myself', 'our', 'doing', 'isn', 'ma', 'o', 'yourself', 's', 'itself', 'ourselves', 'does', 'has', 'my', 'll', 'he', 'own', 'or', 'having', 'mustn', 'at', 'too', 'herself', 'other', 'themselves', 'very', 'don', 'me', 'these', 'with', "she's", 'can', 'is', 'off', 'in', 'to', 'shan', 'those', 'most', 'himself', 'them', 'there', 'ain', 'hasn', 'their', 'nor', 've', 'she', 'was', 'hadn', 'being', 'both', "it's", 'just', 'up', 'as', 'wouldn', 'aren', 'some', 'his', 'we', 'same', 'and', 'more', 'ours', 'because', 'mightn', 'of', 'will', 'do', 'on', 'are', 'no', 'if', "you're", 't', 'about', 'so', 'after', 'few', 'had', 'yourselves', 'while', 'd', 'over', 'this', 'any', 'its', 'once', 'that', 'a', 'again', 'how', 'it', 'who','than', "you'd", 'but', 'until', 'each', 'why', "you'll", 'you', 'from', 'further', 'an', 'through', 'yours', 'have', 'into', 'your', 'should', "mightn't", 'all', 'were', 'by', 're', 'been', 'hers', 'haven', 'him', "that'll", 'during',  'down', 'they', 'out', "should've", 'theirs', 'm', 'the', 'whom', 'when', 'what', 'did', 'her', 'here', 'where', "you've", 'am', "shan't", 'only', 'such', "mustn't", 'then', 'needn', "hadn't", 'weren', 'under', 'i', 'for']

simplefilter(action='ignore', category=FutureWarning)

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
    ans=ans.lower()
    return ans

def avg_feature_vector(sentence, model, num_features, vocab):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in vocab:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

bodys=[]
def extract(headline):
    global bodys
    headline=headline.replace(' ','+')
    link='https://www.google.com/search?q={}&num={}&hl={}'.format(headline,10,'en')
    USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    try:
        page=requests.get(link,headers=USER_AGENT)
        page.raise_for_status()
        soup=BeautifulSoup(page.text,'html.parser')
        bodys=soup.select(".st")
    except:
        bodys=[]

def max_min_func(headline):
    global bodys
    try:
        head=remove_stopwords(headline)
        count=0
        max_res=0
        min_res=2.0
        for body in bodys:
            statement=remove_stopwords(body.get_text())
            s1_afv = avg_feature_vector(head, model=model_word2vec, num_features=200, vocab=vocab)
            s2_afv = avg_feature_vector(statement, model=model_word2vec, num_features=200, vocab=vocab)
            sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
            if math.isnan(sim) != True:
                if sim>max_res:
                        max_res=sim
                if sim<min_res:
                        min_res=sim
            count=count+1
            if count==5:
                break
        return [max_res,min_res]
    except:
        return []

def frequency_bias(headline):
    global bodys
    head=remove_stopwords(headline)
    try:
        count=0
        naive_list=[]
        for body in bodys:
            statement=remove_stopwords(body.get_text())
            naive_list.append(statement)
            count=count+1
            if count==5:
                break
        head_list=head.split(' ')
        scarp_map=dict()
        for statement in naive_list:
            scrap_data=statement.split(' ')
            check=dict()
            for word in scrap_data:
                if word and not check.get(word):
                    check[word]=1
                    if scarp_map.get(word):
                        scarp_map[word]=scarp_map[word]+1
                    else:
                        scarp_map[word]=1
        probality_list=[]
        for word in head_list:
            if scarp_map.get(word):
                probality_list.append(scarp_map[word]/len(naive_list))
            else:
                probality_list.append(0)
        final_probality=sum(probality_list)/len(probality_list)
        for word in head_list:
            try:
                num=int(word)
                if scarp_map.get(word) and scarp_map[word]<=2:
                    final_probality=final_probality-0.1
            except ValueError:
                continue
        return (final_probality)
    except:
        return 0

filename1 = 'finalized_model_svm.sav'
model_svm = joblib.load(filename1)

filename2 = 'finalized_model_LR.sav'
model_lr = joblib.load(filename2)

filename3 = 'finalized_model.sav'
model_final = joblib.load(filename3)

prob1=0
prob2=0
prob3=0
prob4=0
pro1=0
pro2=0
statement=input("Enter the Statement: ")
extract(statement)
max_val=0
min_val=2.0
max_min=max_min_func(statement)
if max_min:
    max_val=max_min[0]
    min_val=max_min[1]
if max_val !=0 and min_val !=2.0:
    prob1=model_svm.predict_proba([[max_val,min_val]])
    pro1=prob1[0][1]
    prob2=model_lr.predict_proba([[max_val,min_val]])
    pro2=prob2[0][1]
prob3=frequency_bias(statement)
print(pro1, pro2 ,prob3)
if pro1 and pro2 and prob3:
    prob4=model_final.predict_proba([[prob3,pro1,pro2]])
    print("statement is true by %", prob4[0][1]*100)
else:
    print("cannot check the fakeness for the following statement ")


