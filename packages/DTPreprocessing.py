#!/usr/bin/python3 
# -*- coding: utf-8 -*-

import pandas as pd 


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

# Encoding target 
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

# Convert emoji 
def convert_emojis(text): 
    '''
        text : string 
    '''

    import emoji
    text = emoji.demojize(text)
    text = text.replace('_','')
    return text.replace(':','')

# Data cleaning 
def data_cleaning(df):
    '''
        df : DataFrame 
    '''
    
    import re 
    import string
   
    
    # Delete IDs
    df.drop('ID', axis=1, inplace=True)
    
    # First encoding 
    df['Label'].replace('none', 'not racist', inplace=True)
    df['Label'].replace('racism', 'racist', inplace=True)
    
    i = 0 
    for i in range(df['Tweets'].shape[0]):
        # Remove ids @ 
        df['Tweets'][i] = re.sub(r'@\S+', '', df['Tweets'][i])
        
        # Remove punctuation
        df['Tweets'][i] = "".join([char for char in df['Tweets'][i] if char not in string.punctuation])
        
        # Uppercase -> Lowercase 
        df['Tweets'][i] = df['Tweets'][i].lower()
        
        # Delete Url 
        df['Tweets'][i] = re.sub(r'http\S+', '', df['Tweets'][i])
        
        # Delete characters 
        df['Tweets'][i] = re.sub("ð|ÿ|‘|œ|¦|€|˜|™|¸|¤|‚|©|¡|…|”|“|‹|š|±|³|iâ|§|„|", '', df['Tweets'][i]) 
        
    return df

# Tokenization 
def tokenization(df):
    '''
        df : DataFrame 
    '''
    
    # Generate tokens
    from nltk.tokenize import TweetTokenizer
    tknz = TweetTokenizer()
    tokens = []
    
    i = 0
    for i in range(df['Label'].shape[0]):
        tokens.extend(tknz.tokenize(df['Tweets'][i]))
    
    return tokens

# Stemming 
def stemming(tokens):
    '''
        tokens : list 
    '''
    
    from nltk.stem import PorterStemmer
    stemming = PorterStemmer()
    for token in tokens:
        token = stemming.stem(token)
    return tokens

# Tokens frequencies 
def tokens_frequencies(tokens):
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
    
    return tokens_freq

# Stop words 
def stop_words(df):
    '''
        df : DataFrame
    '''
    from nltk.corpus import stopwords
    liste = []
    i = 0 
    for i in range(df.shape[0]):
        if df['Tokens'][i] not in stopwords.words('english'):
            liste.append([df['Tokens'][i],df['Frequencies'][i]])
    return pd.DataFrame(liste,columns=["Tokens","Frequencies"])

# Vectorization 
def vectorization(df, nbr_tokens, token_frequency):
    '''
        df : DataFrame 
        nbr_tokens : int - the number of tokens from the token-frequency DataFrame  
        token_frequency : DataFrame - the array that contains the frequency of each token 
    '''
    from nltk.tokenize import TweetTokenizer 

    # Most frequent tokens
    most_freq = token_frequency['Tokens'][:nbr_tokens]

    # Vectorization 
    matrix = []
    for tweet in df['Tweets']:
        vector = []
        tknz = TweetTokenizer()
        tweet = tknz.tokenize(tweet)
        for token in most_freq:
            if token in tweet:
                vector.append(1)
            else:
                vector.append(0)
        matrix.append(vector)
    
    # Convert the matrix into a dataframe
    bag_of_words = pd.DataFrame(matrix, columns=most_freq)
    
    return bag_of_words







