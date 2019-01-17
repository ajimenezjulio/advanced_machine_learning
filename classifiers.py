#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:11:07 2018

@author: juliocesar

"""

# Libraries and Dependencies
# -----------------------------------------------------------------------

from sklearn.svm import LinearSVC #,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from xgboost.sklearn import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier



# Global matrix variable
X = None



# Classifiers
# -----------------------------------------------------------------------
    
# SVM classifier
def svm(data, c):
    # Building the training and test sets
    trx = data['training_data'].toarray()
    trl = data['training_labels'].reshape(X.shape[0],1)
    tsx = data['test_data'].toarray()
    
    # Creating and training classifier 
    classifier = LinearSVC(C = c).fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = classifier.predict(tsx)
    return prediction


# Random Forest Classifier
def rndForest(data, ntrees):
    # Building the training and test sets
    trx = data['training_data'].toarray()
    trl = data['training_labels'].reshape(X.shape[0],1)
    tsx = data['test_data'].toarray()
    
    # Creating and training classifier 
    clf = RandomForestClassifier(n_estimators = ntrees).fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction


# Logistic Regression Classifier
def logReg(data, c):
    # Building the training and test sets
    trx = data['training_data'].toarray()
    trl = data['training_labels'].reshape(X.shape[0],1)
    tsx = data['test_data'].toarray()
    
    # Creating and training classifier
    clf = LogisticRegression(C = c).fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction


# Extreme Gradient Boost Classifier
def xgboost(data, md, ss, cs):
    # Building the training and test sets
    trx = data['training_data'].toarray()
    trl = data['training_labels'].reshape(X.shape[0],1)
    tsx = data['test_data'].toarray()
    
    # Creating and training classifier
    clf = XGBClassifier( max_depth = md, subsample = ss,
                        colsample_bytree = cs).fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction


# Linear Discriminant Analysis Classifier
def lda(data):
    # Building the training and test sets
    trx = data['training_data'].toarray()
    trl = data['training_labels'].reshape(X.shape[0],1)
    tsx = data['test_data'].toarray()
    
    # Creating and training classifier
    clf = LinearDiscriminantAnalysis().fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction


# Quadratic Discriminant Analysis Classifier
def qda(data):
    # Building the training and test sets
    trx = data['training_data'].toarray()
    trl = data['training_labels'].reshape(X.shape[0],1)
    tsx = data['test_data'].toarray()
    
    # Creating and training classifier
    clf = QuadraticDiscriminantAnalysis().fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)  
    return prediction


# Ada Boost Classifier
def adaboost(data, ne):
    # Building the training and test sets
    trx = data['training_data'].toarray()
    trl = data['training_labels'].reshape(X.shape[0],1)
    tsx = data['test_data'].toarray()
    
    # Creating and training classifier
    dt = DecisionTreeClassifier() 
    clf = AdaBoostClassifier(n_estimators = ne, 
                             base_estimator = dt).fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction



# Functions
# -----------------------------------------------------------------------

# Getter for global matrix
def getX():
    global X
    return X


# Setter for global matrix
def setX(value):
    global X
    X = value