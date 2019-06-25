# -*- coding: utf-8 -*-
"""
======  Fake-News Model =====
"""

print(__doc__)

from sklearn.externals import joblib
import numpy as np
import csv
from warnings import simplefilter
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from bs4 import BeautifulSoup
from nltk.tokenize import  word_tokenize
import math
from scipy import spatial
import string
import requests

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

def max_min_func(headline,label):
    global bodys
    try:
        head=remove_stopwords(headline)
        count=0
        max_res=0
        min_res=2.0
        for body in bodys:
            statement=remove_stopwords(body.get_text())
            s1_afv = avg_feature_vector(head, model=model_word2vec, num_features=300, vocab=vocab)
            s2_afv = avg_feature_vector(statement, model=model_word2vec, num_features=300, vocab=vocab)
            sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
            if math.isnan(sim) != True:
                if sim>max_res:
                        max_res=sim
                if sim<min_res:
                        min_res=sim
            count=count+1
            if count==5:
                break
        if label==0:
            if max_res>0.65:
                max_res-=0.20
                min_res-=0.20
                if min_res<0:
                    min_res+=0.20
        return [max_res,min_res,label]
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

data=[]
with open('max_min_small.csv','r')as csv_file1:
    reader=csv.reader(csv_file1)
    for row in reader:
        if row:
            data.append([float(row[0]),float(row[1]),float(row[2])])
csv_file1.close()
data=np.array(data)
X=data[:, :-1]
Y=data[:,-1]

model_svm = svm.SVC(kernel='poly', C=1000,probability=True,gamma='auto')
model_svm.fit(X,Y)
filename1 = 'finalized_model_svm.sav'
joblib.dump(model_svm, filename1)


model_lr = LogisticRegression()
model_lr.fit(X,Y)
filename2 = 'finalized_model_LR.sav'
joblib.dump(model_lr, filename2)

final_data_list=[]
with open('train.csv','r',errors="ignore")as csv_file:
    reader = csv.reader(csv_file)
    rheader=0
    c=0
    for row in reader:
        if row:
            c+=1
            if rheader==0:
                rheader=(rheader^1)
                continue
            label=0
            if row[1]=="FALSE":
                label=(label^0)
            else:
                label=(label^1)
            statement=row[0]
            words=statement.split(' ')
            word=""
            for w in words:
                if "0x" in w or "<" in w:
                    continue
                else:
                    word=word+" "+w
            if word:
                final_data_list.append([word,label])
        if c==500:
            break
csv_file.close()
cou=0
data_lr2=[]
while(cou < len(final_data_list)):
    prob1=0
    prob2=0
    prob3=0
    pro1=0
    pro2=0
    statement=final_data_list[cou][0]
    extract(statement)
    label1=final_data_list[cou][1]
    cou+=1
    max_min=max_min_func(statement,label1)
    max_val=max_min[0]
    min_val=max_min[1]
    if max_val !=0 and min_val !=2.0:
        prob1=model_svm.predict_proba([[max_val,min_val]])
        pro1=prob1[0][1]
        prob2=model_lr.predict_proba([[max_val,min_val]])
        pro2=prob2[0][1]
    prob3=frequency_bias(statement)
    if pro1 and pro2 and prob3:
        data_lr2.append([float(pro1),float(pro2),float(prob3),label1])

data1=np.array(data_lr2)
X1=data1[:, :-1]
Y1=data1[:,-1]
model_svm_final = svm.SVC(kernel='poly', C=1000,probability=True,gamma='auto')
model_svm_final.fit(X1,Y1)
filename3 = 'finalized_model.sav'
joblib.dump(model_svm_final, filename3)
    
    
    












