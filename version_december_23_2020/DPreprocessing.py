#!/usr/bin/python3 
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import spacy
from Vectorizer import vectorization
#from packages.Vectorizer import vectorization
import warnings
warnings.filterwarnings("ignore")

# Balanced sample of the data set 
def balanced_sample(df,nbr_tweets):
    discriminant = df[df['Label']==1.0].sample(int(nbr_tweets/2))
    non_dicriminant = df[df['Label']==0.0].sample(int(nbr_tweets/2))
    return pd.concat([discriminant,non_dicriminant], ignore_index=True)

# Sub-dataframe
def subdataframe(df, n): 
    '''
        Generate a DataFrame of n random examples from the original Dataframe  
        df : Original Dataframe from which a subdataframe is generated 
        n : number of examples (rows) 
    '''

    import random 
    i = random.randint(0,df.shape[0]-n)
    return df.iloc[i:i+n]

def encoding_target(y):
    '''
        y : Series 
    '''
    
    for i in range(y.shape[0]):
        if y[i] == 'sexism' or y[i] == 'racism':
            y[i] = 1
        else:
            y[i] = 0
    return y.astype('int')

def data_cleaning(tweets):
    '''
    Input 
        df : pd.DataFrame 
    Output 
        df : Cleaned pd.DataFrame
    '''
    
    import re 
    import string

    print(">>>>>> Data Cleaning Process : Start")

    # Normalization : Uppercase -> Lowercase 
    tweets = tweets.str.lower()
    print(">>> Normalization ", end="")

    # Remove ids @ 
    tweets = tweets.str.replace(r'@\S+', '', regex=True)
    print("| Remove ids @ ", end="")

    # Replace urls with the tag URL 
    tweets = tweets.str.replace(r'http\S+', '', regex=True)
    print("| Replace urls by URL tag ", end="")


    # Delete special characters 
    tweets = tweets.str.replace(r'ð|ÿ|‘|œ|¦|€|˜|™|¸|¤|‚|©|¡|…|”|“|‹|š|±|³|iâ|§|„|', '', regex=True)
    print("| Remove Special char ", end="")

    # Remove punctuation 
    punctuation_keep = ["!","\'"]
    punctuation_remove = "".join([c for c in string.punctuation if c not in punctuation_keep])
    tweets = tweets.str.replace('[{}]'.format(punctuation_remove),'')
    print("| Remove Punctuation ")



    return tweets 

# Tokenization 
def tokenization(tweets):
    '''
    Input
        tweets : a pd.Series of 1 column 
    Output
        tokens : a list of tokens 
    '''
    
    # Generate tokens
    from nltk.tokenize import TweetTokenizer
    tknz = TweetTokenizer()
    
    tokens = []
    for tweet in tweets:
        tokens.extend(tknz.tokenize(tweet))
    
    return tokens

# Canonization 
def canonization(tweets, method):
    '''
        
    '''
    
    if method == "lemmatization":
        
        import spacy
        nlp = spacy.load('en_core_web_sm')

        Tweets = []
        for tweet in tweets:
            doc = nlp(tweet.strip())
            lemmatized_tokens = []
            for token in doc:
                if token.lemma_ == "-PRON-":
                    lemmatized_tokens.append(str(token).strip())
                else: lemmatized_tokens.append(token.lemma_.strip())
            Tweets.append(" ".join(lemmatized_tokens))

        return pd.Series(Tweets)
    
    elif method == "stemming":
        
        from nltk.stem import PorterStemmer
        stemming = PorterStemmer()

        Tweets = []
        for tweet in tweets:
            stemmed_tokens = []
            for token in tweet.split(): 
                stemmed_tokens.append(stemming.stem(token).strip())
            Tweets.append(" ".join(stemmed_tokens))

        return pd.Series(Tweets)
        
    else : 
        print("There is an error in the chosen method")   

# Most frequent tokens 
def most_frequent_tokens(tokens, nbr_tokens):
    '''
        tokens : list 
    ''' 
    # Creation of a dataframe Tokens-Frequencies
    from nltk.probability import FreqDist
    fdist = FreqDist()
    
    for token in tokens:
        fdist[token] += 1 
    tokens_freq = pd.DataFrame(list(fdist.items()), columns = ["Tokens","Frequencies"])
    
    # Sort the dataframe according to frequency of words
    tokens_freq.sort_values(by='Frequencies',ascending=False, inplace=True)
    
    return tokens_freq['Tokens'].tolist()[:nbr_tokens]

# Stop words 
def stop_words(l):
    '''
        
    '''
    from nltk.corpus import stopwords
    
    liste = []
    for item in l:
        if item not in stopwords.words('english') and len(item) > 1:
            liste.append(item)
    return pd.DataFrame(liste)

# Preprocessing
def preprocessing(dataset, nbr_tokens, vectorizer, canoniz_method):
    '''
        This function aims to combine vectorizing and canonization methods in addition to some data cleaning manipulations. 
        Input

        Output
    '''

    # Copy the dataset
    df = dataset.copy()
    y = df['Label']
    
    print(">>>>>>>>>>>> Preprocessing : Start")
    

    # Data cleaning 
    Tweets = df['Tweets']
    X_cleaned = data_cleaning(Tweets)
    

    # Canonization 
    X_canon = canonization(tweets=X_cleaned, method=canoniz_method)
    print(">>>>>> Canonization done")


    # Vectorization 
    from Vectorizer import vectorization
    X = vectorization(tweets=X_canon, nbr_tokens=nbr_tokens, method=vectorizer)
    print(">>>>>> Vectorization done")
               

    # Split the data : Train set & Test set 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)
    print(">>> Split Data done ")
    print(">>>>>>>>>>>> Preprocessing : End")
    
    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)