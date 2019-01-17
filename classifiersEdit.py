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
from xgboost.sklearn import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler


# Global matrix variable
X = None



# Classifiers
# -----------------------------------------------------------------------
    
# SVM classifier
def svm(trl, trx, tsx, c):
    """
    expecting two df  with labels in col1 and 128 vec embedding
    takes label from first col name of df
    """
    
    # Creating and training classifier 
    classifier = LinearSVC(C = c).fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = classifier.predict(tsx)
    return prediction


# Random Forest Classifier
def rndForest(trl, trx, tsx,ntrees):
    
    # Creating and training classifier 
    clf = RandomForestClassifier(n_estimators = ntrees).fit(trx, trl)
    
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction


# Logistic Regression Classifier
def logReg(trl, trx, tsx, c):

    # Creating and training classifier
    try:
        
        clf = LogisticRegression(C = c).fit(trx, trl.ravel())
    except: #if multiclass..
        clf = LogisticRegression(C = c, solver='newton-cg', multi_class='multinomial').fit(trx, trl.ravel())
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction


# Extreme Gradient Boost Classifier
def xgboost(trl, trx, tsx, md, ss, cs):
    
    # Creating and training classifier
    clf = XGBClassifier( max_depth = md, subsample = ss,
                        colsample_bytree = cs).fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction


# Linear Discriminant Analysis Classifier
def lda(trl, trx, tsx):
    
    # Creating and training classifier
    clf = LinearDiscriminantAnalysis().fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction


# Quadratic Discriminant Analysis Classifier
def qda(trl, trx, tsx):
    # Creating and training classifier
    clf = QuadraticDiscriminantAnalysis().fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)  
    return prediction


# Ada Boost Classifier
def adaboost(trl, trx, tsx, ne):

    # Creating and training classifier
    dt = DecisionTreeClassifier() 
    clf = AdaBoostClassifier(n_estimators = ne, 
                             base_estimator = dt).fit(trx, trl.ravel())
    
    # Making prediction on test set
    prediction = clf.predict(tsx)
    return prediction

def naive_bayes(trl, trx, tsx):
    # Creating and training classifier
    scaler = MinMaxScaler()
    scaler.fit(trx)
    trx = scaler.fit_transform(trx)
    clf = MultinomialNB().fit(trx, trl.ravel())
    prediction = clf.predict(tsx)
    return prediction

def extra_trees(trl, trx, tsx, ne):
    #Creating and training classifier
    clf = ExtraTreesClassifier().fit(trx, trl.ravel())
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
 
classifiers = [svm, rndForest, logReg, xgboost, lda, qda, adaboost]
    
#def expt(classifiers):
#    """
#    """
#    for classifier in classifiers:
#        data = 