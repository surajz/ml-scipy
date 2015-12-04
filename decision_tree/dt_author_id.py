#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
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


print features_train[0]
print "len",len(features_train[0])
#########################################################
### your code goes here ###
from sklearn import  tree

clf = tree.DecisionTreeClassifier(min_samples_split=50)
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)


from sklearn.metrics import accuracy_score
score = accuracy_score(labels_test, pred)
print score

#########################################################


