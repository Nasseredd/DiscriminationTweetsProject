#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Machine Learning Models (Supervised learning)

def support_vector_machine(X_train, X_test, y_train, y_test, metric):
    # training 
    from sklearn.svm import SVC 
    model = SVC(kernel='linear', random_state=0)
    model.fit(X_train, y_train)
    
    # prediction 
    y_pred = model.predict(X_test)
    
    # evaluation 
    from sklearn.metrics import accuracy_score, f1_score
    if metric == "f1score": return f1_score(y_test, y_pred, pos_label='1')
    elif metric == "accuracy": return accuracy_score(y_test, y_pred, pos_label='1')

def logreg(X_train, X_test, y_train, y_test, metric):
    # training 
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # prediction
    y_pred = model.predict(X_test)
    
    # evaluation 
    from sklearn.metrics import accuracy_score, f1_score
    if metric == "f1score": return f1_score(y_test, y_pred)
    elif metric == "accuracy": return accuracy_score(y_test, y_pred)

def random_forest(X_train, X_test, y_train, y_test, metric):
    # training 
    from sklearn.ensemble import RandomForestClassifier 
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    
    # prediction 
    y_pred = model.predict(X_test)
    
    # evaluation 
    from sklearn.metrics import accuracy_score, f1_score
    if metric == "f1score": return f1_score(y_test, y_pred)
    elif metric == "accuracy": return accuracy_score(y_test, y_pred)

def decision_tree(X_train, X_test, y_train, y_test, metric):
    # training 
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    
    # prediction 
    y_pred = model.predict(X_test)
    
    # evaluation 
    from sklearn.metrics import accuracy_score, f1_score
    if metric == "f1score": return f1_score(y_test, y_pred)
    elif metric == "accuracy": return accuracy_score(y_test, y_pred) 

def adaboost(X_train, X_test, y_train, y_test, metric):
    # training 
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(random_state=0)
    model.fit(X_train, y_train)
    
    # prediction 
    y_pred = model.predict(X_test)
    
    # evaluation 
    from sklearn.metrics import accuracy_score, f1_score
    if metric == "f1score": return f1_score(y_test, y_pred)
    elif metric == "accuracy": return accuracy_score(y_test, y_pred)

# Deep Learning Models (Supervised Learning)

def convolution_neural_network(X_train, X_test, y_train, y_test, metric):
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))
    #print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4, verbose=1)
    y_pred = model.predict(X_test)

    # evaluation 
    from sklearn.metrics import accuracy_score, f1_score
    if metric == "f1score": return f1_score(y_test, y_pred.round()) # round because y_pred is a vector of probabilites 
    elif metric == "accuracy": return accuracy_score(y_test, y_pred.round()) # round because y_pred is a vector of probabilites 

def long_short_term_memory(X_train, X_test, y_train, y_test, metric):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(32,1,X_train.shape[1]), return_sequences=False))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4)
    y_pred = model.predict(X_test)
    
    
    # evaluation 
    from sklearn.metrics import accuracy_score, f1_score
    if metric == "f1score": return f1_score(y_test, y_pred.round()) # round because y_pred is a vector of probabilites 
    elif metric == "accuracy": return accuracy_score(y_test, y_pred.round()) # round because y_pred is a vector of probabilites 


