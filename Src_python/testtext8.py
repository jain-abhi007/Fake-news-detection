# -*- coding: utf-8 -*-
"""
===== Final-Usage Model =====
"""
print(__doc__)

from sklearn.externals import joblib
import numpy as np
from warnings import simplefilter
from nltk.tokenize import word_tokenize
import math
from scipy import spatial
import string

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
    print("reached3")
    feature_vec = np.zeros((num_features, ), dtype='float32')
    print(len(feature_vec))
    n_words = 0
    print("reached4")
    for word in words:
        print(word)
        if word in vocab:
            print("reached inside")
            n_words += 1
            print(len(model.wv[word]))
            feature_vec = np.add(feature_vec, model.wv[word])
            print("reached5")
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    print(feature_vec)
    return feature_vec

bodys=['india is a good country','india won world cup in 2011','india won test match against pakistan last night']

def max_min_func():
    global bodys
    try:
        head="india is a good country"
        count=0
        max_res=0
        min_res=2.0
        for body in bodys:
            statement=remove_stopwords(body)
            print("reached1")
            s1_afv = avg_feature_vector(head, model=model_word2vec, num_features=200, vocab=vocab)
            s2_afv = avg_feature_vector(statement, model=model_word2vec, num_features=200, vocab=vocab)
            print("reached2")
            sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
            if math.isnan(sim) != True:
                if sim>max_res:
                        max_res=sim
                if sim<min_res:
                        min_res=sim
            count=count+1
            if count==5:
                break
        print([max_res,min_res])
    except:
        print("error")

max_min_func()

