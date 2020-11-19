#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def bag_of_words(df, nbr_tokens):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=nbr_tokens, 
                                stop_words='english')
    vectorizer.fit(df['Tweets'])
    vocab = list(vectorizer.vocabulary_.keys())
    vector = vectorizer.transform(df['Tweets']).toarray()
    
    # Convert the matrix into a dataframe
    matrix = pd.DataFrame(vector, columns=vocab)
    return matrix

def tfidf(df, nbr_tokens, ngram):
    from tqdm import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer


    tf_idf = TfidfVectorizer(max_features=nbr_tokens,
                             binary=True,
                             smooth_idf=False,
                             min_df=5, 
                             max_df=0.7, 
                             stop_words='english')

    return tf_idf.fit_transform(np.array(df['Tweets'])).toarray()

def hashing(df, nbr_tokens):

    from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer(n_features=nbr_tokens)
    
    return vectorizer.transform(df['Tweets']).toarray()

def word2vec(df, nbr_tokens, tokens):
    
    from gensim.models import Word2Vec

    model = Word2Vec(
            tokens,
            size = 200, 
            #window = 5,                   # context window size
            #min_count = 2,                # Ignores all words with total frequency lower than 2.                                  
            #sg = 1,                       # 1 for skip-gram model
            #hs = 0,
            #negative = 10,                # for negative sampling
            #workers= 32,                  # no.of cores
            seed = 34) 

    model.train(tokens, total_examples= len(df['Tweets']), epochs=10)
    return model 

def vectorization(df, nbr_tokens, tokens, method):
    '''
        
    '''
    
    if method == "bow":
        return bag_of_words(df, nbr_tokens)  

    elif method == "tfidf":
        return tfidf(df, nbr_tokens, ngram=(1,1))
        
    elif method == "hashing":
        return hashing(df, nbr_tokens) 
    
    elif method == "word2vec":
        return word2vec(df, nbr_tokens, tokens) 
