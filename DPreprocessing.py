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

def data_cleaning(df):
    '''
    Input 
        df : pd.DataFrame 
    Output 
        df : Cleaned pd.DataFrame
    '''
    
    import re 
    import string
    
    for tweet in df['Tweets']:
        # Remove ids @ 
        tweet = re.sub(r'@\S+', '', tweet)
    
        # Remove punctuation
        tweet = "".join([char for char in tweet if char not in string.punctuation])
    
        # Uppercase -> Lowercase 
        tweet = tweet.lower()
        
        # Delete Url 
        tweet = re.sub(r'http\S+', '', tweet)
    
        # Delete characters 
        tweet = re.sub(":|\"|ð|ÿ|‘|œ|¦|€|˜|™|¸|¤|‚|©|¡|…|”|“|‹|š|±|³|iâ|§|„|", '', tweet) 

    return df

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
        This is a normalization function which generates a list of (stemmed/lemmatized) tokens as output 
    Input 
        tweets : a pd.Series of 1 column 
        method : a string that refers to the canonization method to choose (stemming or lemmatization)
    Output 
        lemmatized_tokens/stemmed_tokens - a list of stemmed/lemmatized tokens 
    '''
    
    if method == "lemmatization":
        
        # Lemmatization includes in fact Tokenization, Lemmatization and POS 
        # But, it does not need to tokenize tweets before 
        
        import spacy
        nlp = spacy.load('en_core_web_sm')
        
        lemmatized_tokens = []
        for tweet in tweets:
            doc = nlp(tweet)
            for token in doc: 
                if token.lemma_ == "-PRON-":
                    lemmatized_tokens.append(token)
                else: lemmatized_tokens.append(token.lemma_)
                    
        # remove spaces 
        deletables = ['', ' ', '  ', '   ', '    ','     ']
        return [str(item) for item in lemmatized_tokens if str(item) not in deletables]
    
    elif method == "stemming":
        
        # Stemming needs tokenization 
        from nltk.stem import PorterStemmer
        stemming = PorterStemmer()
        
        tokens = tokenization(tweets)
        
        stemmed_tokens = []
        for token in tokens: 
            stemmed_tokens.append(stemming.stem(token))
        return stemmed_tokens
    
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
        if item not in stopwords.words('english'):
            liste.append(item)
    return pd.DataFrame(liste)

# Preprocessing 
def preprocessing(dataset, nbr_tokens, vectorizer, canon):
    '''
        This function aims to combine vectorizing and canonization methods in addition to some data cleaning manipulations. 
        Input 

        Output 
    '''

    from Vectorizer import vectorization

    # Copy the dataset
    df = dataset.copy()
    print(">>> Copy data set")
    y = df['Label']
    
    print(">>> Preprocessing start")
    # preprocessing
    df_cleaned = data_cleaning(df)
    print(">>> Data cleaned ")
    tokens = canonization(df_cleaned['Tweets'], canon)
    print(">>> Canonization done")
    X = vectorization(df_cleaned, nbr_tokens, tokens, method=vectorizer)
    print(">>> Vectorization done")

    #tokens.to_csv("Word-Frenquency.csv")                  

    # Split the data : Train set & Test set 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)
    print(">>> Split Data done ")
    print(">>>>>>>>>>>>>>> End of Preprocessing")
    
    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)
