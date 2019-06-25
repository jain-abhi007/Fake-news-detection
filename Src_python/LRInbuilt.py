# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:37:01 2019

@author: I506768
"""

# Save Model Using joblib
from warnings import simplefilter
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import csv

simplefilter(action='ignore', category=FutureWarning)
data=[]
with open('max_min.csv','r')as csv_file:
    reader=csv.reader(csv_file)
    for row in reader:
        if row:
            data.append([float(row[0]),float(row[1]),float(row[2])])
data=np.array(data)
X=data[:, :-1]
Y=data[:,-1]
print(X)
print(Y)
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model_LR.sav'
joblib.dump(model, filename)
val=model.predict_proba(X_test)
print(val)
# some time later...

# load the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, Y_test)
#print(result)