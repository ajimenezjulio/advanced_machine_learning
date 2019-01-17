#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 00:16:38 2018

@author: juliocesar

DG comments:    Confirm files[3] is correct for test data rather than files[2]
                Should use validation approach to standardise methodology
"""

# Libraries and Dependencies
# -----------------------------------------------------------------------

# import numpy as np
import pandas as pd
import scipy.io
import glob
import classifiers as clf
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------
# Training files: 5172 
# Test files:     5857 
# ----------------------


def main():
    # Getting Data
    # --------------------------------------------------------------------
    
    print("\n\nLoading Data...")
    # Files Path
    DIR = 'data/'
    files = [DIR+"spam/", DIR+"ham/", DIR+"test/", DIR+"enron2/ham/"]
    
    # Get all files for each set
    spam = glob.glob( files[0] + '*.txt')
    ham = glob.glob(  files[1] + '*.txt')
    test = glob.glob( files[3] + '*.txt')
    
    # Joining spam and ham to form the training set 
    spham = spam + ham
    
    # Construct vectorizer for TF-IDF feature extraction 
    vectorizer = TfidfVectorizer(input='filename',lowercase = True, stop_words="english",
                                 encoding='latin-1', min_df = 8)
    
    
    # Get features
    tr = vectorizer.fit_transform(spham)
    ts = vectorizer.transform(test)
    
    # Creating matrices of data and labels for training set 
    X = tr
    clf.setX(X)
    Y = [1] * len(spam) + [0] * len(ham)
    
    # Dictionary pointing to the different sets
    dicy = {}
    dicy['training_data'] = X
    dicy['training_labels'] = Y
    dicy['test_data'] = ts
    
    # Saving dictionary as variable to reuse if necessary
    scipy.io.savemat('data.mat', dicy)
    
    
    
    # Classifying
    # -------------------------------------------------------------------
    
    print("Classifying...")
    # Loading data
    data = scipy.io.loadmat('./data.mat') 
    
    # SVM
    # C = Penalty parameter of the error term (Try different values > 0)
    C = 0.1
    res_svm = clf.svm(data, C)
    
    # Random Forest
    # ntrees = The number of trees in the forest
    ntrees = 100
    res_rforest = clf.rndForest(data, ntrees)
    
    # Logistic Regression
    # C = Inverse of regularization strength
    C = 1.0
    res_lg = clf.logReg(data, C)
    
    # Extreme Gradient Boost
    # md = Max depth
    # ss = Subsample ratio of training instance
    # cs = Subsample ratio of columns when constructing each tree
    md = 1
    ss = 0.8
    cs = 0.8
    res_xgb = clf.xgboost(data, md, ss, cs)
    
    # Linear Discriminant Analysis
    res_lda = clf.lda(data)
    
    # Quadratic Discriminant Analysis
    res_qda = clf.qda(data)
    
    # Ada Boost
    # ne = Maximum number of estimators at which boosting is terminated
    ne = 50
    res_ada = clf.adaboost(data, ne)
    
    # Building array of results
    results = [res_svm, res_rforest, res_lg, res_xgb, res_lda, res_qda, res_ada]
    
    print("Writing Results...")
    # Saving results
    createCSV(res_svm, 'resSVM')
    createCSV(res_rforest, 'resRndForest')
    createCSV(res_lg, 'resLogReg')
    createCSV(res_xgb, 'resXGBoost')
    createCSV(res_lda, 'resLDA')
    createCSV(res_qda, 'resQDA')
    createCSV(res_ada, 'resADA')

    # 0 as second parameter to evaluat 'Ham' accuracy
    printRes(results, 0) 
    print(len(res_svm))
    print("\n ****** Finished ******")


# Functions
# -----------------------------------------------------------------------

# Create a CSV file containing the classifier results
def createCSV(data, title):
    # Adding data to the frame
    df = pd.DataFrame(data)
    # Create Id column, value = actual index
    df['Id'] = df.index
    # Assign names to columns
    df.columns = ['Category', 'Id']
    # Building file
    df.to_csv(title + '.csv', header = True, columns=['Id','Category'], index = False)


# Printing results in console
def printRes(results, spam = 1):
    methods = ["SVM:\t\t", "RndForest:\t", "LogReg:\t", "XGradientBoost:", "LDA:\t\t", "QDA:\t\t", "ADABoost:\t"]
    
    if spam == 1:
        print("----- Spam Accuracy -----")
    else:
        print("----- Ham Accuracy ------")
    
    i = 0
    for result in results:
        # Calculating accuracy for spam or ham emails
        if spam == 1:
            accu = sum(result) / len(result)
        else:
            accu = - (sum(result) / len(result)) + 1
            
        print('\n', methods[i], accu )
        i += 1
        
        
        
# Execute Program
# ----------------------
if __name__ == '__main__':
    main()