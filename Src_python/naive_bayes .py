# -*- coding: utf-8 -*-
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
stopWords=['be','which', 'y', 'myself', 'our', 'doing', 'isn', 'ma', 'o', 'yourself', 's', 'itself', 'ourselves', 'does', 'has', 'my', 'll', 'he', 'own', 'or', 'having', 'mustn', 'at', 'too', 'herself', 'other', 'themselves', 'very', 'don', 'me', 'these', 'with', "she's", 'can', 'is', 'off', 'in', 'to', 'shan', 'those', 'most', 'himself', 'them', 'there', 'ain', 'hasn', 'their', 'nor', 've', 'she', 'was', 'hadn', 'being', 'both', "it's", 'just', 'up', 'as', 'wouldn', 'aren', 'some', 'his', 'we', 'same', 'and', 'more', 'ours', 'because', 'mightn', 'of', 'will', 'do', 'on', 'are', 'no', 'if', "you're", 't', 'about', 'so', 'after', 'few', 'had', 'yourselves', 'while', 'd', 'over', 'this', 'any', 'its', 'once', 'that', 'a', 'again', 'how', 'it', 'who','than', "you'd", 'but', 'until', 'each', 'why', "you'll", 'you', 'from', 'further', 'an', 'through', 'yours', 'have', 'into', 'your', 'should', "mightn't", 'all', 'were', 'by', 're', 'been', 'hers', 'haven', 'him', "that'll", 'during',  'down', 'they', 'out', "should've", 'theirs', 'm', 'the', 'whom', 'when', 'what', 'did', 'her', 'here', 'where', "you've", 'am', "shan't", 'only', 'such', "mustn't", 'then', 'needn', "hadn't", 'weren', 'under', 'i', 'for']
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


def frequency_bias(headline):
    headline=remove_stopwords(headline)
    head=remove_stopwords(headline)
    headline=headline.replace(' ','+')
    link='https://www.google.com/search?q={}&num={}&hl={}'.format(headline,10,'en')
    USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    try:
        page=requests.get(link,headers=USER_AGENT)
        page.raise_for_status()
        soup=BeautifulSoup(page.text,'html.parser')
        bodys=soup.select(".st")
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
        print(final_probality*100)
    except:
        print("0%")
headline=input("enter the statement: ")
frequency_bias(headline)


