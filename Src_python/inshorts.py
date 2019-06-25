# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:29:55 2019

@author: I506768
"""

# -*- coding: utf-8 -*-
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import csv
from random import shuffle
from sklearn.externals import joblib

filename='word_to_vect.sav'
model=joblib.load(filename)
vocab=list(model.vocab)
stopWords=['be','which', 'y', 'myself', 'our', 'doing', 'isn', 'ma', 'o', 'yourself', 's', 'itself', 'ourselves', 'does', 'has', 'my', 'll', 'he', 'own', 'or', 'having', 'mustn', 'at', 'too', 'herself', 'other', 'themselves', 'very', 'don', 'me', 'these', 'with', "she's", 'can', 'is', 'off', 'in', 'to', 'shan', 'those', 'most', 'himself', 'them', 'there', 'ain', 'hasn', 'their', 'nor', 've', 'she', 'was', 'hadn', 'being', 'both', "it's", 'just', 'up', 'as', 'wouldn', 'aren', 'some', 'his', 'we', 'same', 'and', 'more', 'ours', 'because', 'mightn', 'of', 'will', 'do', 'on', 'are', 'no', 'if', "you're", 't', 'about', 'so', 'after', 'few', 'had', 'yourselves', 'while', 'd', 'over', 'this', 'any', 'its', 'once', 'that', 'a', 'again', 'how', 'it', 'who','than', "you'd", 'but', 'until', 'each', 'why', "you'll", 'you', 'from', 'further', 'an', 'through', 'yours', 'have', 'into', 'your', 'should', "mightn't", 'all', 'were', 'by', 're', 'been', 'hers', 'haven', 'him', "that'll", 'during',  'down', 'they', 'out', "should've", 'theirs', 'm', 'the', 'whom', 'when', 'what', 'did', 'her', 'here', 'where', "you've", 'am', "shan't", 'only', 'such', "mustn't", 'then', 'needn', "hadn't", 'weren', 'under', 'i', 'for']
final_list=[]
c=0

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

#first dataset
    
fa=0
with open('inshorts_data.csv','r',errors='ignore') as csv_file:
    reader = csv.reader(csv_file)
    rheader=0
    for row in reader:
        if row:
            if rheader<40000:
                rheader+=1
                continue
            statement=row[0]
            words=statement.split(' ')
            word=""
            for w in words:
                if "0x" in w or "<" in w:
                    continue
                else:
                    word=word+" "+w
            if fa < 2000:
                fake=fakeit(word)
                if fake != 0:
                    final_list.append([word,0])
                    fa+=1
                else:
                    final_list.append([word,1])
            else:
                final_list.append([word,1])
            c+=1
        if c==8000:
            break
csv_file.close()

shuffle(final_list)
with open('fakeit_inshorts6.csv','a')as csv_file:
    writer=csv.writer(csv_file)
    for i in range(0,len(final_list)):
        writer.writerow(final_list[i])
csv_file.close()