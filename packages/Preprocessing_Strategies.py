#!/usr/bin/python3 
# -*- coding: utf-8 -*-

from packages.DTPreprocessing import * 
import pandas as pd 


# Strategy 1 
def preprocessing_1(dataset, nbr_tokens):
    '''
        dataset : DataFrame - the raw data set 
        nbr_tokens : int - the number of tokens from the token-frequency DataFrame 
        nbr_tweets : int - the number of tweets to vectorize 
    '''

    # Copy the dataset
    df = dataset.copy()
    y = df['Label']
    
    # manipulations
    df_cleaned = data_cleaning(df)
    print("data cleaning end")
    
    # tokenization
    #tokens = tokenization(df_cleaned)
    
    # stemming
    #tokens_stemmed = stemming(tokens)
    
    # tokens_frequencies 
    #tokfreq = tokens_frequencies(tokens_stemmed)
    
    # Stop words 
    #tokfreq = stop_words(tokfreq)
    
    # Generate a CSV file for Tokens-Frequencies
    #tokfreq.to_csv("Word-Frenquency.csv")
    
    # vectorization
    #X = vectorization(df, nbr_tokens, tokfreq)
    
    # Encoding target 
    #y = encoding_target(y)

    #print("X.shape = ", X.shape," et y.shape = ", y.shape)
    
    # Split the data : Train set & Test set 
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)
    

    return 1,2,3,4
