b  # -*- coding: utf-8 -*-
import re
import urllib.error
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.externals import joblib
import csv
import math
import numpy as np
from scipy import spatial
import string
import requests


filename='word_to_vect.sav'
model=joblib.load(filename)
vocab=list(model.vocab)
stopWords=['be','which', 'y', 'myself', 'our', 'doing', 'isn', 'ma', 'o', 'yourself', 's', 'itself', 'ourselves', 'does', 'has', 'my', 'll', 'he', 'own', 'or', 'having', 'mustn', 'at', 'too', 'herself', 'other', 'themselves', 'very', 'don', 'me', 'these', 'with', "she's", 'can', 'is', 'off', 'in', 'to', 'shan', 'those', 'most', 'himself', 'them', 'there', 'ain', 'hasn', 'their', 'nor', 've', 'she', 'was', 'hadn', 'being', 'both', "it's", 'just', 'up', 'as', 'wouldn', 'aren', 'some', 'his', 'we', 'same', 'and', 'more', 'ours', 'because', 'mightn', 'of', 'will', 'do', 'on', 'are', 'no', 'if', "you're", 't', 'about', 'so', 'after', 'few', 'had', 'yourselves', 'while', 'd', 'over', 'this', 'any', 'its', 'once', 'that', 'a', 'again', 'how', 'it', 'who','than', "you'd", 'but', 'until', 'each', 'why', "you'll", 'you', 'from', 'further', 'an', 'through', 'yours', 'have', 'into', 'your', 'should', "mightn't", 'all', 'were', 'by', 're', 'been', 'hers', 'haven', 'him', "that'll", 'during',  'down', 'they', 'out', "should've", 'theirs', 'm', 'the', 'whom', 'when', 'what', 'did', 'her', 'here', 'where', "you've", 'am', "shan't", 'only', 'such', "mustn't", 'then', 'needn', "hadn't", 'weren', 'under', 'i', 'for']
fake_real_data=[]
finals_data_list=[]

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


def extract(headline,label):
    head=headline
    headline=headline.replace(' ','+')
    link='https://www.google.com/search?q={}&num={}&hl={}'.format(headline,10,'en')
    USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    print(head)
    try:
        page=requests.get(link,headers=USER_AGENT)
        page.raise_for_status()
        soup=BeautifulSoup(page.text,'html.parser')
        bodys=soup.select(".st")
        count=0
        max_res=0
        min_res=2.0
        for body in bodys:
            statement=remove_stopwords(body.get_text())
            s1_afv = avg_feature_vector(head, model=model, num_features=300, vocab=vocab)
            s2_afv = avg_feature_vector(statement, model=model, num_features=300, vocab=vocab)
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
            if max_res>6.5:
                max_res-=2.0
                min_res-=2.0
                if min_res<0:
                    min_res+=2.0
        print(max_res,min_res,label)
        return [max_res,min_res,label]
    except:
        print("error")
        return []
        
        

def valid_entry_check(word):
        """
        Check if input is null or contains only spaces or numbers or special characters
        """
        temp = re.sub(r'[^A-Za-z ]', ' ', word)
        temp = re.sub(r"\s+", " ", temp)
        temp = temp.strip()
        if temp != "":
            return True
        return False


def fakeit(sentence):
    global stopWords
    count=0
    l=list(sentence.split(' '))
    for i,val in enumerate(l):
        if val not in stopWords:
            antonym_list=[]
            for syn in wordnet.synsets(val): 
                for li in syn.lemmas(): 
                    if li.antonyms():
                        for ant in li.antonyms():
                            word=ant.name()
                            antonym_list.append(word)
            for antonym in antonym_list:
                if antonym in vocab:
                    l[i]=antonym
                    count+=1
                    break
    if count==0:
        return 0
    return ' '.join(l)


with open('getdata.csv','r')as csv_file:
    reader = csv.reader(csv_file)
    count=0
    for row in reader:
        if row:
            if count%2==0:
                count=count+1
                fake_data=fakeit(row[0])
                if fake_data==0:
                    fake_real_data.append([row[0],1])
                    count=count-1
                else:
                    fake_real_data.append([fake_data,0])
            else:
                count=count+1
                fake_real_data.append([row[0],1])
csv_file.close()
            

with open('fake_real_data.csv','w')as csv_file:
    writer = csv.writer(csv_file)
    for i in range(0,len(fake_real_data)):
        writer.writerow(fake_real_data[i])
csv_file.close()


with open('fake_real_data.csv','r')as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        if row:
            final_data=extract(row[0],row[1])
            if final_data:
                if final_data[0] != 0 and final_data[1] != 2.0:
                    finals_data_list.append(final_data)
csv_file.close()


with open('final_dataset.csv','w')as csv_file:
    writer = csv.writer(csv_file)
    for i in range(0,len(finals_data_list)):
        writer.writerow(finals_data_list[i])
csv_file.close()
            
    
