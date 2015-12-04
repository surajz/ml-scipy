#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf =  svm.SVC(kernel='rbf', C=10000.0)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "training time:", round(time() - t0, 3),"s"

print '10',pred[10]
print '26',pred[26]
print '50',pred[50]

count = 0
for p in pred:
    if p == 0:
        count = count+1

print "count",count        

from sklearn.metrics import accuracy_score
score = accuracy_score(labels_test, pred)
print score
#########################################################


