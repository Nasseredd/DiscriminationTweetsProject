#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def bag_of_words(tweets, nbr_tokens):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=nbr_tokens, 
                                stop_words='english')
    vectorizer.fit(tweets)
    vocab = list(vectorizer.vocabulary_.keys())
    vector = vectorizer.transform(tweets).toarray()
    
    # Convert the matrix into a dataframe
    matrix = pd.DataFrame(vector, columns=vocab)
    return matrix

def tfidf(tweets, nbr_tokens, ngram):
    from tqdm import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer


    tf_idf = TfidfVectorizer(max_features=nbr_tokens,
                             binary=True,
                             smooth_idf=False,
                             min_df=5, 
                             max_df=0.7, 
                             stop_words='english')

    return tf_idf.fit_transform(np.array(tweets)).toarray()

def hashing(tweets, nbr_tokens):

    from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer(n_features=nbr_tokens)
    
    return vectorizer.transform(tweets).toarray()

def word2vec(tweets, nbr_tokens, tokens):
    
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

    model.train(tokens, total_examples= len(tweets), epochs=10)
    return model 

def vectorization(tweets, nbr_tokens, method):
    '''
        
    '''
    
    if method == "bow":
        return bag_of_words(tweets, nbr_tokens)  

    elif method == "tfidf":
        return tfidf(tweets, nbr_tokens, ngram=(1,1))
        
    elif method == "hashing":
        return hashing(tweets, nbr_tokens) 
    
    #elif method == "word2vec":
        #return word2vec(tweets, nbr_tokens, tokens) 
