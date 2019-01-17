#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:40:24 2018

@author: juliocesar
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix


def DeepNeural_w2v(feat_train, y_train, x_test, feat_test, idmodel, vocab, epoch = 1):
    
    """
    # x_train and x_test are list of list of word tokens
    
    w2vec = MeanEmbeddingVectorizer(w2v)
    feat_train = w2vec.transform(x_train)
    feat_test = w2vec.transform(x_test)
    
    
    """
    
    print("Padding sequences in vectors...")
    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 10000
    # truncate and pad input sequences
    max_review_length = 100
    
    X_train = sequence.pad_sequences(feat_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(feat_test, maxlen=max_review_length)
    
    print("Building DNN model...")
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=epoch, batch_size=64)
    
    print("Saving model files...")
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("modelCNN" + idmodel + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modelCNN" + idmodel + ".h5")
    print("Saved")
    
    print("---------------------------- Training ------------------------------------")
    prediction = model.predict_classes(X_train)
    cm = confusion_matrix(prediction, y_train)
    scores = model.evaluate(X_train, y_train, verbose=0)
    print(cm)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print("------------------------------ Test --------------------------------------")
    prediction = model.predict_classes(X_test)
    cm = confusion_matrix(prediction, y_test)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(cm)
    print("Accuracy: %.2f%%" % (scores[1]*100))