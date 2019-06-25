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
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [0,1]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)
simplefilter(action='ignore', category=FutureWarning)
data=[]
with open('final_max_min_prob3.csv','r')as csv_file1:
    reader=csv.reader(csv_file1)
    for row in reader:
        if row:
            data.append([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
shuffle(data)
csv_file1.close()
data=np.array(data)
X=data[:, 1:-1]
Y=data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

print("######### ACCURACY MEASURES FOR SVM #########")
model_svm = svm.SVC(kernel='poly', C=1000,probability=True,gamma='auto')
model_svm.fit(X_train,y_train)
predicted = model_svm.predict(X_test)
print("ACCURACY--",accuracy_score(y_test, predicted))
print("CONFUSION MATRIX --",confusion_matrix(y_test, predicted))
#print("CLASSIFICATION REPORT",classification_report(y_test, predicted))
plot_confusion_matrix(y_test,predicted,classes=[1,0],title='CONFUSION MATRIX FOR SVM')
#filename1 = 'finalized_model_svm2.sav'
#joblib.dump(model_svm, filename1)

print("######### ACCURACY MEASURES FOR LR #########")
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)
predicted = model_lr.predict(X_test)
print("ACCURACY--",accuracy_score(y_test, predicted))
print("CONFUSION MATRIX --",confusion_matrix(y_test, predicted))
#print("CLASSIFICATION REPORT",classification_report(y_test, predicted))
plot_confusion_matrix(y_test,predicted,classes=[1,0],title='CONFUSION MATRIX FOR LR')
#filename2 = 'finalized_model_LR1.sav'
#joblib.dump(model_lr, filename2)


final_train=[]
for row in data:
    max_val=row[1]
    min_val=row[2]
    pro_svm=model_svm.predict_proba([[max_val,min_val]])
    pro1=pro_svm[0][1]
    pro_lr=model_lr.predict_proba([[max_val,min_val]])
    pro2=pro_lr[0][1]
    pro3=row[0]
    final_train.append([pro1,pro2,pro3,row[3]])
  
final_train=np.array(final_train)
X1=final_train[:, :-1]
Y1=final_train[:,-1]

model_svm_final = svm.SVC(kernel='poly', C=1000,probability=True,gamma='auto')
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.25)
print("######### ACCURACY MEASURES FOR SVM FINAL MODEL#########")
model_svm_final = svm.SVC(kernel='poly', C=1000,probability=True,gamma='auto')
model_svm_final.fit(X_train,y_train)
predicted = model_svm_final.predict(X_test)
print("ACCURACY--",accuracy_score(y_test, predicted))
print("CONFUSION MATRIX --",confusion_matrix(y_test, predicted))
print("CLASSIFICATION REPORT",classification_report(y_test, predicted))
plot_confusion_matrix(y_test,predicted,classes=[1,0],title='CONFUSION MATRIX FOR FINAL MODEL')
#filename3 = 'finalized_model1.sav'
#joblib.dump(model_svm_final, filename3)
